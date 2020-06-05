import json
import os
import os.path as osp
import random

import cv2
import imgaug as ia
import numpy as np
import torch
import torch.nn.functional as F
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from pytorch_modules.utils import IMG_EXT

POINTS_WH = 30

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

TRAIN_AUGS = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(
            iaa.CropAndPad(
                percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2)
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2)
                },  # translate by -20 to +20 percent (per axis)
                rotate=(-90, 90),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[
                    0,
                    1
                ],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(
                    0,
                    255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.
                ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(
            (0, 5),
            [
                sometimes(
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
                ),  # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur(
                        (0,
                         3.0)),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(
                        k=(2, 7)
                    ),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(
                        k=(3, 11)
                    ),  # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0),
                            lightness=(0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.BlendAlphaSimplexNoise(
                    iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                               direction=(0.0, 1.0)),
                    ])),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255),
                    per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5
                                ),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15),
                                      size_percent=(0.02, 0.05),
                                      per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                iaa.Add(
                    (-10, 10), per_channel=0.5
                ),  # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation(
                    (-20, 20)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.BlendAlphaFrequencyNoise(
                        exponent=(-4, 0),
                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                        background=iaa.LinearContrast((0.5, 2.0)))
                ]),
                iaa.LinearContrast(
                    (0.5, 2.0),
                    per_channel=0.5),  # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),  # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(
                    0.01, 0.05))),  # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True)
    ],
    random_order=True)


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, augments, multi_scale, rect, with_label,
                 mosaic):
        super(BasicDataset, self).__init__()
        self.img_size = img_size
        self.rect = rect
        self.multi_scale = multi_scale
        self.augments = augments
        self.with_label = with_label
        self.mosaic = mosaic
        self.data = []

    def get_data(self, idx):
        return None, None

    def __getitem__(self, idx):
        img, bboxes = self.get_item(idx)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        x1 = np.float32([min(label[2::2]) for label in bboxes])
        x2 = np.float32([max(label[2::2]) for label in bboxes])
        y1 = np.float32([min(label[3::2]) for label in bboxes])
        y2 = np.float32([max(label[3::2]) for label in bboxes])

        xywh = np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
                        1)
        cid = np.float32([label[:2]
                          for label in bboxes]) if len(bboxes) else np.zeros(
                              (0, 2))
        bboxes = np.concatenate([cid, xywh], 1)
        img, bboxes = torch.ByteTensor(img), torch.FloatTensor(bboxes)
        return img, bboxes

    def get_item(self, idx, mosaic=True):
        img, polygons = self.get_data(idx)
        img = img[..., ::-1]
        h, w, c = img.shape

        if self.rect:
            scale = min(self.img_size[0] / w, self.img_size[1] / h)
            resize = iaa.Sequential([
                iaa.Resize({
                    'width': int(w * scale),
                    'height': int(h * scale)
                }),
                iaa.PadToFixedSize(*self.img_size,
                                   pad_cval=[123.675, 116.28, 103.53],
                                   position='center')
            ])
        else:
            resize = iaa.Resize({
                'width': self.img_size[0],
                'height': self.img_size[1]
            })

        img = resize.augment_image(img)
        polygons = resize.augment_polygons(polygons)
        # augment
        if self.augments is not None:
            if (self.mosaic
                    and random.random() < 0.3) or (not self.mosaic
                                                   and random.random() < 0.5):
                augments = self.augments.to_deterministic()
                img = augments.augment_image(img)
                polygons = augments.augment_polygons(polygons)

        labels = []
        for polygon in polygons.polygons:
            p = polygon.exterior.reshape(-1, 2)
            p[:, 0] /= img.shape[1]
            p[:, 1] /= img.shape[0]
            # p = p.clip(0, 1)
            if p.min() < 0 or p.max() > 1:
                continue
            c = polygon.label
            labels.append([0, c] + p.reshape(-1).tolist())
        if self.mosaic and mosaic and random.random() < 0.2:
            mosaic_list = [(img, labels)]
            for _ in range(3):
                next_idx = random.randint(0, self.__len__() - 1)
                mosaic_list.append(self.get_item(next_idx, False))
            cut_x_ratio = random.uniform(0.2, 0.8)
            cut_y_ratio = random.uniform(0.2, 0.8)
            cut_x = int(cut_x_ratio * self.img_size[0])
            cut_y = int(cut_y_ratio * self.img_size[1])
            random.shuffle(mosaic_list)
            img0 = mosaic_list[0][0][:cut_y, :cut_x]
            img1 = mosaic_list[1][0][:cut_y, cut_x:]
            img0 = np.concatenate([img0, img1], 1)
            img2 = mosaic_list[2][0][cut_y:, :cut_x]
            img3 = mosaic_list[3][0][cut_y:, cut_x:]
            img2 = np.concatenate([img2, img3], 1)
            img = np.concatenate([img0, img2], 0)
            labels0 = mosaic_list[0][1]
            labels0 = [
                label for label in labels0 if max(label[2::2]) < cut_x_ratio
            ]
            labels0 = [
                label for label in labels0 if max(label[3::2]) < cut_y_ratio
            ]
            labels1 = mosaic_list[1][1]
            labels1 = [
                label for label in labels1 if min(label[2::2]) > cut_x_ratio
            ]
            labels1 = [
                label for label in labels1 if max(label[3::2]) < cut_y_ratio
            ]
            labels2 = mosaic_list[2][1]
            labels2 = [
                label for label in labels2 if max(label[2::2]) < cut_x_ratio
            ]
            labels2 = [
                label for label in labels2 if min(label[3::2]) > cut_y_ratio
            ]
            labels3 = mosaic_list[3][1]
            labels3 = [
                label for label in labels3 if min(label[2::2]) > cut_x_ratio
            ]
            labels3 = [
                label for label in labels3 if min(label[3::2]) > cut_y_ratio
            ]
            # labels = np.concatenate([labels0, labels1, labels2, labels3], 0)
            labels = labels0 + labels1 + labels2 + labels3
        # filter small items
        labels = [
            label for label in labels
            if max(label[::2]) - min(label[2::2]) > 3e-3
        ]
        labels = [
            label for label in labels
            if max(label[::2]) - min(label[3::2]) > 3e-3
        ]
        return img, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        imgs, dets = list(zip(*batch))  # transposed

        for i, l in enumerate(dets):
            l[:, 0] = i  # add target image index for build_targets()
        dets = torch.cat(dets, 0)
        imgs = torch.stack(imgs, 0)
        return imgs, dets

    def post_fetch_fn(self, batch):
        imgs, labels = batch
        imgs = imgs.float()
        imgs -= torch.FloatTensor([123.675, 116.28,
                                   103.53]).reshape(1, 3, 1, 1).to(imgs.device)
        imgs /= torch.FloatTensor([58.395, 57.12,
                                   57.375]).reshape(1, 3, 1, 1).to(imgs.device)

        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 32) * 32
            w = int(w * scale / 32) * 32
            imgs = F.interpolate(imgs, (h, w))
        return (imgs, labels)


class YOLODataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False,
                 with_label=True,
                 mosaic=False):
        super(YOLODataset, self).__init__(img_size=img_size,
                                          augments=augments,
                                          multi_scale=multi_scale,
                                          rect=rect,
                                          with_label=with_label,
                                          mosaic=mosaic)
        self.path = path
        self.classes = []
        self.build_data()
        self.data.sort()

    def build_data(self):
        data_dir = osp.dirname(self.path)
        with open(osp.join(data_dir, 'classes.names'), 'r') as f:
            self.classes = f.readlines()
        image_dir = osp.join(data_dir, 'images')
        label_dir = osp.join(data_dir, 'labels')
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        names = [name for name in names if osp.splitext(name)[1] in IMG_EXT]
        for name in names:
            bboxes = []
            label_name = osp.join(label_dir, osp.splitext(name)[0] + '.txt')
            if osp.exists(label_name):
                with open(label_name, 'r') as f:
                    lines = [[float(x) for x in l.split(' ') if x]
                             for l in f.readlines() if l]

                for l in lines:
                    if len(l) != 5:
                        continue
                    c = l[0]
                    x = l[1]
                    y = l[2]
                    w = l[3] / 2.
                    h = l[4] / 2.
                    xmin = x - w
                    xmax = x + w
                    ymin = y - h
                    ymax = y + h
                    if ymax > 1 or xmax > 1 or ymin > 1 or xmin > 1:
                        continue
                    if ymax < 0 or xmax < 0 or ymin < 0 or xmin < 0:
                        continue
                    if ymax <= ymin or xmax <= xmin:
                        continue
                    bboxes.append([c, xmin, ymin, xmax, ymax])
            self.data.append([osp.join(image_dir, name), bboxes])
            if self.with_label:
                self.data = [d for d in self.data if len(d[1]) > 0]

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        polygons = []
        for c, xmin, ymin, xmax, ymax in self.data[idx][1]:
            polygons.append(
                Polygon(
                    np.float32(
                        [xmin, ymin, xmin, ymax, xmax, ymax, xmax,
                         ymin]).reshape(-1, 2), c))
        polygons = PolygonsOnImage(polygons, img.shape)


class CocoDataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False,
                 with_label=False,
                 mosaic=False):
        super(CocoDataset, self).__init__(img_size=img_size,
                                          augments=augments,
                                          multi_scale=multi_scale,
                                          rect=rect,
                                          with_label=with_label,
                                          mosaic=mosaic)
        with open(path, 'r') as f:
            self.coco = json.loads(f.read())
        self.img_root = osp.dirname(path)
        self.augments = augments
        self.classes = []
        self.build_data()
        self.data.sort()

    def build_data(self):
        img_ids = []
        img_paths = []
        img_anns = []
        self.classes = [c['name'] for c in self.coco['categories']]
        for img_info in self.coco['images']:
            img_ids.append(img_info['id'])
            img_paths.append(osp.join(self.img_root, img_info['file_name']))
            img_anns.append([])
        for ann in self.coco['annotations']:
            idx = ann['image_id']
            idx = img_ids.index(idx)
            img_anns[idx].append(ann)
        self.data = list(zip(img_paths, img_anns))
        if self.with_label:
            self.data = [d for d in self.data if len(d[1]) > 0]

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        anns = self.data[idx][1]
        polygons = []
        for ann in anns:
            if len(ann['segmentation'][0]) > 4:
                polygons.append(
                    Polygon(
                        np.float32(ann['segmentation']).reshape(-1, 2),
                        ann['category_id']))
            elif len(ann['segmentation'][0]) == 2:
                # point
                polygons.append(
                    Polygon(
                        np.float32([
                            ann['segmentation'][0][0] - POINTS_WH / 2,
                            ann['segmentation'][0][1] - POINTS_WH / 2,
                            ann['segmentation'][0][0] - POINTS_WH / 2,
                            ann['segmentation'][0][1] + POINTS_WH / 2,
                            ann['segmentation'][0][0] + POINTS_WH / 2,
                            ann['segmentation'][0][1] + POINTS_WH / 2,
                            ann['segmentation'][0][0] + POINTS_WH / 2,
                            ann['segmentation'][0][1] - POINTS_WH / 2
                        ]).reshape(-1, 2), ann['category_id']))
            else:
                print(len(ann['segmentation'][0]))
        polygons = PolygonsOnImage(polygons, img.shape)
        return img, polygons

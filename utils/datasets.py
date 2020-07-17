import json
import os
import os.path as osp
import random
from collections import defaultdict

import cv2
import imgaug as ia
import numpy as np
import torch
import torch.nn.functional as F
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from pytorch_modules.utils import IMG_EXT

POINTS_WH = 60

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
                rotate=(-15, 15),  # rotate by -45 to +45 degrees
                shear=(-8, 8),  # shear by -16 to +16 degrees
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
        sometimes(
            iaa.OneOf([
                iaa.GaussianBlur(
                    (0, 3.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(
                    k=(2, 7)
                ),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(
                    k=(3, 11)
                ),  # blur image using local medians with kernel sizes between 2 and 7
            ])),
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
        img, bboxes = torch.FloatTensor(img), torch.FloatTensor(bboxes)
        return img, bboxes

    def get_item(self, idx):
        img, polygons = self.get_data(idx)
        img = img[..., ::-1]
        h, w, c = img.shape

        # resize
        if self.rect:
            scale = min(self.img_size[0] / w, self.img_size[1] / h)
            resize = iaa.Sequential([
                iaa.Resize({
                    'width': int(w * scale),
                    'height': int(h * scale)
                }),
                iaa.PadToFixedSize(*self.img_size,
                                   pad_cval=0,
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
            augments = self.augments.to_deterministic()
            img = augments.augment_image(img)
            polygons = augments.augment_polygons(polygons)

        # post processing
        h, w, c = img.shape
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # record all instance points and category
        instance_list = defaultdict(list)
        for polygon in polygons.polygons:
            p = polygon.exterior.reshape(-1, 2)
            p[:, 0] /= w
            p[:, 1] /= h

            # clip polygon
            p = p.clip(0, 1)

            # # or remove polygon out of image
            # if p.min() < 0 or p.max() > 1:
            #     continue

            category = polygon.label['category_id']
            instance_id = polygon.label['instance_id']
            instance_list[instance_id].append((p, category))

        bboxes = []
        for instance in instance_list.values():
            category = instance[0][1]
            points = np.concatenate(list(map(lambda x: x[0], instance)))
            x1 = points[:, 0].min()
            y1 = points[:, 1].min()
            x2 = points[:, 0].max()
            y2 = points[:, 1].max()
            x = (x1 + x2) / 2.
            y = (y1 + y2) / 2.
            w = x2 - x1
            h = y2 - y1
            bboxes.append([0, category, x, y, w, h])
        if len(bboxes):
            bboxes = np.float32(bboxes)
        else:
            bboxes = np.zeros((0, 6), dtype=np.float32)
        return img, bboxes

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        imgs, dets = list(zip(*batch))  # transposed
        imgs = torch.stack(imgs, 0)

        imgs -= torch.FloatTensor([123.675, 116.28,
                                   103.53]).reshape(1, 3, 1, 1)
        imgs /= torch.FloatTensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 32) * 32
            w = int(w * scale / 32) * 32
            imgs = F.interpolate(imgs, (h, w))

        for i, l in enumerate(dets):
            l[:, 0] = i  # add target image index for build_targets()
        dets = torch.cat(dets, 0)
        return imgs, dets


class YOLODataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=[416, 416],
                 augments=None,
                 multi_scale=[],
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
                 img_root=None,
                 img_size=[416, 416],
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
        print('json loaded')
        if img_root is None:
            self.img_root = osp.dirname(path)
        else:
            self.img_root = img_root
        self.augments = augments
        self.classes = []
        self.build_data()

    def build_data(self):
        self.classes = []
        self.classes_hash = dict()
        for idx, category in enumerate(self.coco['categories']):
            self.classes.append(category['name'])
            self.classes_hash[category['id']] = idx

        img_paths = dict()
        img_anns = defaultdict(list)
        for img_info in self.coco['images']:
            img_paths[img_info['id']] = osp.join(self.img_root,
                                                 img_info['file_name'])
        for ann in self.coco['annotations']:
            # not supported
            if ann['iscrowd'] == 1:
                continue
            img_anns[ann['image_id']].append(ann)
        self.data = []
        for idx, img_path in img_paths.items():
            ann = img_anns[idx]
            if self.with_label and len(ann) == 0:
                continue
            self.data.append((img_path, ann))
        # self.data = self.data[:100]

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        anns = self.data[idx][1]
        polygons = []
        for ann_id, ann in enumerate(anns):
            for seg in ann['segmentation']:
                if len(seg) > 4:
                    polygons.append(
                        Polygon(
                            np.float32(seg).reshape(-1, 2), {
                                'category_id':
                                self.classes_hash[ann['category_id']],
                                'instance_id':
                                ann_id
                            }))
                elif len(seg) == 2:
                    # point
                    polygons.append(
                        Polygon(
                            np.float32([
                                seg[0] - POINTS_WH / 2, seg[1] - POINTS_WH / 2,
                                seg[0] - POINTS_WH / 2, seg[1] + POINTS_WH / 2,
                                seg[0] + POINTS_WH / 2, seg[1] + POINTS_WH / 2,
                                seg[0] + POINTS_WH / 2, seg[1] - POINTS_WH / 2
                            ]).reshape(-1, 2), {
                                'category_id':
                                self.classes_hash[ann['category_id']],
                                'instance_id':
                                ann_id
                            }))
                else:
                    print('segmentation with 2 points is not supported: {}'.
                          format(seg))
        polygons = PolygonsOnImage(polygons, img.shape)
        return img, polygons

import argparse
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import YOLOV3
from pytorch_modules.utils import Fetcher, device
from utils.datasets import CocoDataset
from utils.utils import (ap_per_class, bbox_iou, clip_coords, compute_loss,
                         non_max_suppression, show_batch, xywh2xyxy)


@torch.no_grad()
def test(model, fetcher, conf_thres=1e-3, nms_thres=0.5):
    model.eval()
    val_loss = 0
    classes = fetcher.loader.dataset.classes
    num_classes = len(classes)
    seen = 0
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP',
                                 'F1')
    p, r, f1, mp, mr, mAP, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(enumerate(fetcher), total=len(fetcher))
    for idx, (imgs, targets) in pbar:
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        val_loss += compute_loss(train_out, targets,
                                 model).item()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out,
                                     conf_thres=conf_thres,
                                     nms_thres=nms_thres)
        # Plot images with bounding boxes
        if idx == 0:
            show_batch(imgs, output)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > 0.5 and m[
                            bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu().numpy(), pred[:, 6].cpu().numpy(), tcls))
        pbar.set_description('loss: %8g' % (val_loss / (idx + 1)))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]

    # sync stats
    if dist.is_initialized():
        for i in range(len(stats)):
            stat = torch.FloatTensor(stats[i]).to(device)
            ls = torch.IntTensor([len(stat)]).to(device)
            ls_list = [
                torch.IntTensor([0]).to(device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(ls_list, ls)
            ls_list = [ls_item.item() for ls_item in ls_list]
            max_ls = max(ls_list)
            if len(stat) < max_ls:
                stat = torch.cat(
                    [stat, torch.zeros(max_ls - len(stat)).to(device)])
            stat_list = [
                torch.zeros(max_ls).to(device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(stat_list, stat)
            stat_list = [
                stat_list[si][:ls_list[si]]
                for si in range(dist.get_world_size()) if ls_list[si] > 0
            ]
            stat = torch.cat(stat_list)
            stats[i] = stat.cpu().numpy()

    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, mAP, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, mAP, mf1))

    # Print results per class
    for i, c in enumerate(ap_class):
        print(pf % (classes[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
    # Return results
    mAPs = np.zeros(num_classes) + mAP
    for i, c in enumerate(ap_class):
        mAPs[c] = ap[i]
    # return (mp, mr, mAP, mf1, *(loss / len(dataloader)).tolist()), mAPs
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('val', type=str)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('-s',
                        '--img_size',
                        type=int,
                        nargs=2,
                        default=[416, 416])
    parser.add_argument('-bs', '--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.1,
                        help='object confidence threshold')
    parser.add_argument('--nms-thres',
                        type=float,
                        default=0.5,
                        help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()

    val_data = CocoDataset(opt.val,
                          img_size=opt.img_size,
                          augments=None,
                          rect=opt.rect)
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        collate_fn=CocoDataset.collate_fn,
    )
    val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)
    model = YOLOV3(len(val_data.classes))
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location='cpu')
        model.load_state_dict(state_dict['model'])
    metrics = test(model,
                   val_fetcher,
                   conf_thres=opt.conf_thres,
                   nms_thres=opt.nms_thres)
    print('metrics: %8g' % (metrics))

import itertools
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.engine import BasicLitModel

from .utils import (ap_per_class, bbox_iou, clip_coords, compute_loss,
                    non_max_suppression, show_batch, xywh2xyxy)


class YOLOV3(BasicLitModel):
    """
    YOLOv3 object detection model
    """

    def __init__(self, cfg):
        super(YOLOV3, self).__init__(cfg)

    def forward(self, x):
        output = super(YOLOV3, self).forward(x)
        if self.training:
            return tuple(output)
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        # separate losses
        lbox, lobj, lcls = compute_loss(outputs, targets, self.model)
        lbox *= 3.54
        lobj *= 64.3
        lcls *= 37.4
        # total loss
        loss = lbox + lobj + lcls

        return {
            'loss': loss,
            'log': {
                'obj_loss': lobj,
                'cls_loss': lcls,
                'giou_loss': lbox
            }
        }

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        _, _, height, width = images.shape
        inference_outputs, train_outputs = self(images)
        # separate losses
        lbox, lobj, lcls = compute_loss(train_outputs, targets, self.model)
        lbox *= 3.54
        lobj *= 64.3
        lcls *= 37.4
        # total loss
        loss = lbox + lobj + lcls

        bboxes = non_max_suppression(inference_outputs,
                                     conf_thres=0.001,
                                     nms_thres=0.5)
        # Statistics per image
        stats = []
        seen = 0
        for si, pred in enumerate(bboxes):
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
            stats.append(
                (correct, pred[:,
                               4].cpu().numpy(), pred[:,
                                                      6].cpu().numpy(), tcls))
        return {'stats': stats, 'val_loss': loss, 'seen': seen}

    def validation_epoch_end(self, val_outs):
        classes = self.val_dataset.classes
        num_classes = len(classes)

        val_loss = list(map(lambda x: x['val_loss'], val_outs))
        if len(val_loss):
            val_loss = sum(val_loss) / len(val_loss)
        else:
            val_loss = 0

        seen = sum(map(lambda x: x['seen'], val_outs))

        # Compute statistics
        stats = list(
            itertools.chain(*list(map(lambda x: x['stats'], val_outs))))
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]
        # sync stats
        if dist.is_initialized():
            for i in range(len(stats)):
                stat = torch.FloatTensor(stats[i]).to(self.device)
                ls = torch.IntTensor([len(stat)]).to(self.device)
                ls_list = [
                    torch.IntTensor([0]).to(self.device)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(ls_list, ls)
                ls_list = [ls_item.item() for ls_item in ls_list]
                max_ls = max(ls_list)
                if len(stat) < max_ls:
                    stat = torch.cat([
                        stat,
                        torch.zeros(max_ls - len(stat)).to(self.device)
                    ])
                stat_list = [
                    torch.zeros(max_ls).to(self.device)
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
            nt = np.bincount(
                stats[3].astype(np.int64),
                minlength=num_classes)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, mAP, mf1))

        # Print results per class
        for i, c in enumerate(ap_class):
            print(pf % (classes[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
        return {
            'mAP': torch.FloatTensor([mAP]),
            'val_loss': val_loss,
        }

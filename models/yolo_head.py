import torch
import torch.nn as nn
from pytorch_modules.nn.utils import ConvNormAct


class YOLOHead(nn.Module):
    def __init__(self, in_feature_id, in_channels, anchors, num_classes,
                 stride):
        super(YOLOHead, self).__init__()
        self.in_feature_id = in_feature_id
        self.conv = nn.Sequential(
            ConvNormAct(in_channels, in_channels),
            nn.Conv2d(in_channels,
                      len(anchors) * (5 + num_classes), 1))
        self.anchors = torch.Tensor(anchors)
        self.stride = stride
        self.na = len(anchors)  # number of anchors (3)
        self.nc = num_classes  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.stride

    def forward(self, p):
        p = p[self.in_feature_id]
        p = self.conv(p)
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny,
                   self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(
                io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            torch.sigmoid_(io[..., 4:])

            if self.nc == 1:
                io[...,
                   5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p

    def create_grids(self, ng=(13, 13), device='cpu', type=torch.float32):
        nx, ny = ng  # x and y grid size

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view(
            (1, 1, ny, nx, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1,
                                              2).to(device).type(type)
        self.ng = torch.Tensor(ng).to(device)
        self.nx = nx
        self.ny = ny

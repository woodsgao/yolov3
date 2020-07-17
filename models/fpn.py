import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.nn.utils import ConvNormAct, SeparableConvNormAct


class FPN(nn.Module):
    def __init__(self,
                 in_features,
                 in_channels,
                 out_channels=[128, 256, 512],
                 reps=3):
        """
        
        Arguments:
            in_features (list): index of the input features you need.
            in_channels (list): channels of input features.
        
        Keyword Arguments:
            out_channels (list): channels_output features  (default: {[128, 256, 512]})
            reps (int) -- repeat times (default: {3})
        """
        super(FPN, self).__init__()
        self.in_features = in_features
        self.fpn_list = nn.ModuleList([])
        self.trans = nn.ModuleList([])
        self.intrans = nn.ModuleList([])
        last_channels = 0
        in_channels = list(in_channels)
        in_channels.reverse()
        out_channels = list(out_channels)
        out_channels.reverse()
        for i in range(len(in_channels)):
            if i > 0:
                self.trans.append(
                    ConvNormAct(last_channels, out_channels[i], 1))
                last_channels = out_channels[i]
            self.intrans.append(ConvNormAct(in_channels[i], out_channels[i], 1))
            last_channels += out_channels[i]
            fpn = [ConvNormAct(last_channels, out_channels[i], 1)]
            fpn += [
                ConvNormAct(out_channels[i], out_channels[i])
                for _ in range(reps)
            ]
            fpn = nn.Sequential(*fpn)
            self.fpn_list.append(fpn)
            last_channels = out_channels[i]

    def forward(self, features):
        features = [features[i] for i in self.in_features]
        features.reverse()
        features = [self.intrans[i](f) for i, f in enumerate(features)]
        new_features = []
        for i in range(len(self.fpn_list)):
            feature = features[i]
            if len(new_features):
                last_feature = self.trans[i - 1](new_features[-1])
                last_feature = F.interpolate(last_feature,
                                             scale_factor=2,
                                             mode='bilinear',
                                             align_corners=False)
                feature = torch.cat([last_feature, feature], 1)
            feature = self.fpn_list[i](feature)
            new_features.append(feature)
        new_features.reverse()
        return new_features

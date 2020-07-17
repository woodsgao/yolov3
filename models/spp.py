import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, in_feature_id=-1, sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.in_feature_id = in_feature_id
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(size, 1, padding=(size - 1) // 2) for size in sizes])

    def forward(self, in_features):
        x = in_features[self.in_feature_id]
        feature_list = [x]
        for pool in self.pools:
            feature_list.append(pool(x))
        feature = torch.cat(feature_list, 1)
        in_features[self.in_feature_id] = feature
        return in_features

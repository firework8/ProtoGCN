import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import math
import numpy as np
import torch.nn.functional as F

from ..builder import HEADS
from .base import *


@HEADS.register_module()
class SimpleHead(BaseHead):

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 in_channels,
                 weight,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 mode='GCN',
                 **kwargs):
        super().__init__(joint_cfg, num_classes, in_channels, weight, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import linalg as LA
from abc import ABCMeta, abstractmethod
from ...core import top_k_accuracy
from ..builder import build_loss

from ..losses.Class_Specific_Contrastive_Loss import Class_Specific_Contrastive_Loss


class BaseHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 in_channels,
                 weight,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.weight = weight
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        if joint_cfg == 'nturgb+d':     # 25*25=625
            n_channel = 625
        elif joint_cfg == 'coco_new':   # 20*20=400
            n_channel = 400
        self.csc_loss = Class_Specific_Contrastive_Loss(num_classes, n_channel)
        
    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, get_graph, label, **kwargs):

        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        """
        ************
        *** Loss ***
        ************
        """  
        loss_cls_1 = self.loss_cls(cls_score, label, **kwargs)
        loss_cls_2 = self.csc_loss(get_graph, label.detach(), cls_score.detach())        
        loss_cls = loss_cls_1.mean() + self.weight * loss_cls_2.mean()
        
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses

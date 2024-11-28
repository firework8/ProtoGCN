import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np


"""
*******************************************
*** Class-Specific Contrastive Learning ***
*******************************************
"""  
class Class_Specific_Contrastive_Loss(nn.Module):
    
    def __init__(self, n_class, n_channel=625, h_channel=256, tmp=0.125, mom=0.9, pred_threshold=0.0):
        super(Class_Specific_Contrastive_Loss, self).__init__()
        self.n_channel = n_channel
        self.h_channel = h_channel
        self.n_class = n_class
        self.tmp = tmp
        self.mom = mom
        self.pred_threshold = pred_threshold
        self.avg_f = torch.randn(self.h_channel, self.n_class)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        
    def onehot(self, label):
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class)
        
        return ones.view(*size).float()

    def get_mask(self, lbl_one, pred_one, logit):
        # 16 120
        tp = lbl_one * pred_one
        # 16 120
        tp = tp * (logit > self.pred_threshold).float()
        
        return tp

    def local_average(self, f, mask):
        b, k = mask.size()
        # 256 16
        f = f.permute(1, 0)
        # 256 120
        avg_f = self.avg_f.detach().to(f.device)

        # 16 120 -> 1 120
        mask_sum = mask.sum(0, keepdim=True)
        # 256 16 * 16 120 -> 256 120
        f_mask = torch.matmul(f, mask)
        f_mask = f_mask / (mask_sum + 1e-12)
        
        # 1 120
        has_object = (mask_sum > 1e-8).float()
        
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        # 256 120
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem

        return f_mem

    def get_score(self, feature, lbl_one, f_mem):
        # 16 256
        (b, c), k = feature.size(), self.n_class
        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)
        
        # 120 256
        f_mem = f_mem.permute(1, 0)
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)

        # 120 16 = 120 256 * 256 16
        score_mem = torch.matmul(f_mem, feature.permute(1, 0))
        score_cl = score_mem / self.tmp

        return score_cl
    
    def forward(self, feature, lbl, logit):
        # batch: 16  num_class: 120
        # 16 256
        feature = self.cl_fc(feature)
        # 16 120 -> 16
        pred = logit.max(1)[1]
        # 16 120
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)
        # 16 120
        logit = torch.softmax(logit, 1)
        
        mask = self.get_mask(lbl_one, pred_one, logit)
        f_mem = self.local_average(feature, mask)
        score_cl = self.get_score(feature, lbl_one, f_mem)
        # 16 120
        score_cl = score_cl.permute(1, 0).contiguous()
        
        return self.loss(score_cl, lbl).mean()

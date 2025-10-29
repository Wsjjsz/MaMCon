#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 上午9:35
# @Author  : Shujia Wei
# @File    : loss.py

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_true, y_pred, mask=None):
        eps = torch.finfo(y_pred.dtype).eps
        y_pred = torch.clamp(y_pred, eps, 1. - eps)
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
        loss = self.alpha * torch.pow(1. - pt_1, self.gamma) * torch.log(pt_1) + (1 - self.alpha) * torch.pow(pt_0, self.gamma) * torch.log(1. - pt_0)
        if not mask == None:
            loss = -torch.sum(loss * mask)
        else:
            loss = -torch.sum(loss)
        return loss


#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 上午10:17
# @Author  : Shujia Wei
# @File    : MaMC.py


import os, sys
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
import haiku as hk
import jax
""" Sequences 1D to 2D """
def seq_1Dto2D(input1D):
    device = input1D.device
    bz, c, L = input1D.size()
    ### expand dim3
    out1 = input1D.unsqueeze(3).to(device)
    repeat_idx = [1] * out1.dim()
    repeat_idx[3] = L
    out1 = out1.repeat(*(repeat_idx))
    ### expand dim2
    out2 = input1D.unsqueeze(2).to(device)
    repeat_idx = [1] * out2.dim()
    repeat_idx[2] = L
    out2 = out2.repeat(*(repeat_idx))
    return torch.cat([out1, out2], dim=1)


class Mamba_layer(nn.Module):
    def __init__(self, in_channels, channels=256):
        super(Mamba_layer, self).__init__()

        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.linear1 = nn.Linear(in_channels, channels)
        self.linear2 = nn.Linear(in_channels, channels)
        self.sig = nn.Sigmoid()
        self.mamba = Mamba2(  # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=channels,  # Model dimension d_model
                d_state=128,  # SSM state expansion factor, typically 64 or 128
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        self.linear3 = nn.Linear(channels, in_channels)
        self.act = nn.SiLU()
    def forward(self, x):
        if x.dim() == 3:
            x_ = x.permute(0, 2, 1)
        elif x.dim() == 4:
            b, c, l, _ = x.shape
            x_ = x.view(b, c, -1).permute(0, 2, 1)
        x_m = self.linear1(x_)
        x_s = self.linear2(x_)
        x_m = self.mamba(x_m)
        gate = self.sig(x_m)
        o = x_s * gate
        o = self.linear3(o)
        if x.dim() == 3:
            o = o.permute(0, 2, 1)
            x = self.conv1d(x)
        elif x.dim() == 4:
            b, c, l, _ = x.shape
            o = o.permute(0, 2, 1).view(b, -1, l, l)
            x = self.conv2d(x)
        o = o + x
        return o

""" Triangle multiplication update """
class TriangleMultiplication(nn.Module):
    def __init__(self, model_args):
        super(TriangleMultiplication, self).__init__()
        self.dz = model_args['Channel_z']  # 64
        self.dc = model_args['Channel_c']  # 64
        
        # init norm
        self.ln1 = nn.LayerNorm(self.dz)
        self.ln2 = nn.LayerNorm(self.dz)
        self.ln3 = nn.LayerNorm(self.dz)
        #line
        self.l1 = nn.Linear(self.dz, self.dc)
        self.l2 = nn.Linear(self.dz, self.dc)
        self.l3 = nn.Linear(self.dz, self.dc)
        self.l4 = nn.Linear(self.dz, self.dc)
        # gate
        self.g1 = nn.Linear(self.dz, self.dc)
        self.g2 = nn.Linear(self.dz, self.dc)
        self.g3 = nn.Linear(self.dz, self.dc)
        self.g4 = nn.Linear(self.dz, self.dc)
        # final output
        self.ln5 = nn.LayerNorm(self.dc)
        self.l5 = nn.Linear(self.dc, self.dz)
        self.g5 = nn.Linear(self.dc, self.dz)
        self.ElU = nn.ELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, z1, z2, z3):
        """
        Argument:
            z1 : (B, C, L, L)
            z2 : (B, C, L, L)
            z3 : (B, C, L, L)
        return:
            z : (B, C, L, L)
        """
        z1 = self.ln1(z1.permute(0, 2, 3, 1))
        z2 = self.ln2(z2.permute(0, 2, 3, 1))
        z3 = self.ln3(z3.permute(0, 2, 3, 1))
        r = z1
        z12 = self.l1(z1) * (self.g1(z1).sigmoid())
        z13 = self.l2(z1) * (self.g2(z1).sigmoid())
        z2 = self.l3(z2) * self.g3(z2).sigmoid()
        z3 = self.l4(z3) * self.g4(z3).sigmoid()
        z12 = torch.einsum(f"bikc,bkjc->bijc", z2, z12)
        z13 = torch.einsum(f"bikc,bjkc->bjic", z3, z13)
        z = z12 + z13
        z = self.g5(r).sigmoid() * self.l5(self.ln5(z))
        z = self.ElU(z)
        z = self.drop(z.permute(0, 3, 1, 2))
        return z


""" TriangleSelfAttention """
class TriangleSelfAttention(nn.Module):
    def __init__(self, model_args):
        super(TriangleSelfAttention, self).__init__()

        self.dz = model_args['Channel_z']  # 64
        self.dc = model_args['Channel_c']  # 8
        self.num_head = model_args['num_head']  # 4
        self.dhc = self.num_head * self.dc  # 32

        self.ln1 = nn.LayerNorm(self.dz)
        self.l_q = nn.Linear(self.dz, self.dhc)
        self.l_k = nn.Linear(self.dz, self.dhc)
        self.l_v = nn.Linear(self.dz, self.dhc)
        # self.Linear_bias = nn.Linear(self.dz, self.num_head)
        self.g1 = nn.Linear(self.dz, self.dhc)
        self.l2 = nn.Linear(self.dhc, self.dz)

    def reshape_dim(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.dc)
        return x.view(*new_shape)

    def forward(self, z, eps=5e4):
        """
        z: (B, C, L, L)
        return: (B, C, L, L)
        """
        z = self.ln1(z.permute(0, 2, 3, 1))
        scalar = torch.sqrt(torch.tensor(1.0 / self.dc))
        q = self.reshape_dim(self.l_q(z))
        k = self.reshape_dim(self.l_k(z))
        v = self.reshape_dim(self.l_v(z))
        attn = torch.einsum(f"bnihc, bnjhc->bhnij", q * scalar, k)
        attn_w = F.softmax(attn, -1)
        o = torch.einsum(f"bhnij, bnjhc->bnihc", attn_w, v)
        g = (self.reshape_dim(self.g1(z))).sigmoid()
        z = (o * g).contiguous().view(o.size()[:-2] + (-1,))
        z = (self.l2(z)).permute(0, 3, 1, 2)
        return z

class MaMC(nn.Module):
    """
        MaMC model
    """

    def __init__(self, args):
        super(MaMC, self).__init__()
        # reduce the high-dimensional 1D input
        self.conv1d_1 = nn.Conv1d(args['InChannels_1D'], args['Channels_1D'], kernel_size=1)
        self.conv1d_2 = nn.Conv1d(args['Channels_1D'], args['OutChannels_1D'], args['Kernel_size_1D'],padding="same")
        self.conv2d_1 = nn.Conv2d(args['InChannels_2D'], args['Channels_2D'], kernel_size=1)
        self.conv2d_2 = nn.Conv2d(args['Channels_2D'], 1, args['Kernel_size_out'], padding="same")
        # layer1d for Mamba-2
        layer1d = [Mamba_layer(args['Channels_1D'],args['mamba_channel']) for i in range(args['mamba1d_layers'])]
        self.layer1 = nn.Sequential(*layer1d)
        # layer2d for Mamba-2
        layer2d = [Mamba_layer(args['Channels_2D'],args['mamba_channel']) for i in range(args['mamba2d_layers'])]
        self.layer2 = nn.Sequential(*layer2d)
        
        self.Triangle_layer = args["triangle_layer"]
        self.Triangle = TriangleMultiplication(args["triangle"])

        
        layer_attention = [TriangleSelfAttention(args["attention"]) for i in range(args["attention_layers"])]
        self.attention = nn.Sequential(*layer_attention)
        
        # output
        self.bn1 = nn.BatchNorm2d(args['Channels_2D'], momentum=0.01, eps=1e-3, affine=True,
                                  track_running_stats=False)

        self.elu1 = nn.ELU()
        self.sig = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
        

    def forward(self, input1D, input2D, mask=None):

        input1D = self.conv1d_1(input1D)
        out1D = self.layer1(input1D)
        out1D = self.conv1d_2(out1D)
        seq2D = seq_1Dto2D(out1D)
        cat2D = torch.cat([seq2D, input2D], dim=1)
        out2D = self.conv2d_1(cat2D)
        Tri2D_C = out2D
        out2D = self.layer2(out2D)
        out2D = self.bn1(out2D)
        out2D = self.elu1(out2D)
        for idx in range(self.Triangle_layer):
            out2D = self.Triangle(out2D, Tri2D_C, out2D)
        out2D = self.attention(out2D)
        out = self.conv2d_2(out2D)
        out = self.sig(out)
        return out

    
    


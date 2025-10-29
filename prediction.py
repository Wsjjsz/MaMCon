#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 上午9:38
# @Author  : Shujia Wei
# @File    : prediction.py

import os
import argparse
import time
from networks.MaMCon import MaMC
from dataset.dataset import train_valid_test_pdb_list, train_valid_test_pdb_file
from utils import *
import torch
import pandas as pd
from loss.loss import FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-wd', type=float, default=1e-2, help='weight decay')
parser.add_argument('--pdb_list_path', type=str, default="./data/", help='relative path')
parser.add_argument('--file_list_path', type=str, default="/data", help='absolute path')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('--threshold', type=int, default=8)
parser.add_argument('--gap', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mask', type=int, default=1)
parser.add_argument('--is_feature_1D', type=str, default="True, True, True",
                    help='1d_embedding, DSSP_ACC, DSSP')
parser.add_argument('--dim_1D', type=int, default=768 + 1 + 8, help='1d_embedding, PSSM, DSSP_ACC,DSSP')
parser.add_argument('--is_feature_2D', type=str, default="True, True, True",
                    help='row_attention, docking, AF2_Mon')
parser.add_argument('--dim_2D', type=int, default=144 + 1 + 1,
                    help='row_attention, Mon_distance, docking')
parser.add_argument('--Truncate_length', type=int, default=400)
parser.add_argument('--Truncate_random', type=bool, default=True)
parser.add_argument('--mamba1d_layers', type=int, default=3)
parser.add_argument('--mamba2d_layers', type=int, default=1)
parser.add_argument('--triangle_layer', type=int, default=7)
parser.add_argument('--attention_layers', type=int, default=2)
parser.add_argument('--OutChannels_1D', type=int, default=2)
parser.add_argument('--metrics', type=str, default=[1, 10, 25, 50, "L/10", "L/5", "L/2", "L/1", 100, "cov"])
parser.add_argument('--Validate_metrics', type=str, default="100")
parser.add_argument('--log', type=str,
                    default="example_test")  #
parser.add_argument('--best_epoch', type=int, default=38)


args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


models_args = {
    "mamba1d_layers": args.mamba1d_layers,  # 3
    "InChannels_1D": args.dim_1D,
    "Channels_1D": 32,
    "OutChannels_1D": args.OutChannels_1D,
    "Kernel_size_1D": 17,
    "mamba2d_layers": args.mamba2d_layers,  # 1
    "InChannels_2D": args.dim_2D + 2 * args.OutChannels_1D,
    "Channels_2D": 64,
    "Kernel_size_2D": 3,
    "Kernel_size_out": 3,
    "mamba_channel":256,
    "triangle_layer": args.triangle_layer,# 7
    "triangle": {
        "Channel_z": 64,
        "Channel_c": 64
    },
    "attention_layers": args.attention_layers,  # 2
    "attention": {
        "Channel_z": 64,
        "Channel_c": 8,
        "num_head": 4
    },

}

data_path_train, data_path_valid, data_path_test, data_path_test_list_218, data_path_CASP_CAPRI_test,data_path_test_AF2,data_path_test_list_218_AF2,data_path_CASP_CAPRI_test_AF2 = train_valid_test_pdb_file(
    args.file_list_path)
train_list, valid_list, test_list, test_list_218, CASP_CAPRI_test_list,test_list_AF2,test_list_218_AF2,CASP_CAPRI_test_list_AF2= train_valid_test_pdb_list(args.pdb_list_path)

model = MaMC(models_args).to(device)

FileName = "./outputs/example/" + args.log
f = open(FileName + ".txt", 'w')

print("---------- start train ----------")
model.load_state_dict(torch.load('./outputs/model/%d_epoch_model.pth' % (args.best_epoch)))
eval_list = ["2QUD", "2IA0"]
path_eval_file = "./example/"
i = 0
for batch in iterate_minibatches(eval_list, path_eval_file, args, shuffle=False):
    inputs_1d, inputs_2d, targets, masks = batch
    input1D, input2D, target, masks = CUDA_selected(inputs_1d), CUDA_selected(inputs_2d), CUDA_selected(
        targets), CUDA_selected(masks.unsqueeze(dim=1))
    y_pred = model(input1D, input2D)
    target = target.squeeze()
    target = target < args.threshold
    prediction = np.array(y_pred.detach().cpu()).squeeze()
    prediction = (prediction + prediction.transpose((1, 0))) / 2.0
    prediction = CUDA_selected(torch.tensor(prediction))
    np.savetxt(f'./outputs/example/outputs_{eval_list[i]}.txt', prediction.squeeze().detach().cpu().numpy())
    metric = TopAccuracy(prediction.squeeze(), target, masks.squeeze(), top=args.metrics, gap=args.gap)
    plot_contacts_and_predictions(prediction,target, eval_list[i], metric["L/1"], "./outputs/example/" + f"{eval_list[i]}.png", "MaMCon")
    print("  ----------- {}  ------------".format(eval_list[i]))
    f.write("  ----------- {}  ------------\n".format(eval_list[i]))
    for item in metric.keys():
        print("  test {} accuracy:\t\t{:.1f}%".format(item, metric[item] * 100))
        f.write("  test {} accuracy:\t\t{:.1f}%\n".format(item, metric[item] * 100))
    print("")
    f.write("\n")
    i = i + 1

print("----------- end train ----------")
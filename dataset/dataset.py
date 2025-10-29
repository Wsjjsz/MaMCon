#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 上午9:06
# @Author  : Shujia Wei
# @File    : dataset.py

import pickle as pkl
import os
import numpy as np
import copy
import torch


def read_data_from_file(pdb_list, data_path, args):
    """
    Arguments:
        pdb_list  : List of protein ids
        data_path : Protein file path
        args : parameters
            :arg.is_feature_1D "True,True,True" or [True,True,True]
            :arg.is_feature_2D "True,True,True" or [True,True,True]
    Return:
        data_all  : data for Dataset (1D, 2D features, Mask, Target)
    """
    if isinstance(args.is_feature_1D, str):
        # Working with String Types
        is_feature_1D = [x.strip() == "True" for x in args.is_feature_1D.split(',')]
    elif isinstance(args.is_feature_1D, list):
        # Work with list types (if needed)
        is_feature_1D = args.is_feature_1D

    if isinstance(args.is_feature_2D, str):
        # Working with String Types
        is_feature_2D = [x.strip() == "True" for x in args.is_feature_2D.split(',')]
    elif isinstance(args.is_feature_2D, list):
        # Work with list types (if needed)
        is_feature_2D = args.is_feature_2D


    data_all = []
    for pdb in pdb_list:

        # load the preprocess data
        with open(data_path + "/" + pdb + ".pkl", 'rb') as f:
            data = pkl.load(f, encoding='iso-8859-1')
        # input features of 1D
        seq_data = None

        if is_feature_1D[2]:
            seq_data = data["DSSP"] if seq_data is None else np.concatenate((seq_data, data["DSSP"]), axis=1)
        if is_feature_1D[1]:
            seq_data = data["DSSP_ACC"] if seq_data is None else np.concatenate((seq_data, data["DSSP_ACC"]), axis=1)
        if is_feature_1D[0]:
            seq_data = data['esm_msa_1d'] if seq_data is None else np.concatenate((seq_data, data['esm_msa_1d']),
                                                                                  axis=1)

        # input features of 2D

        esm_row_attentions = data['row_attentions']
        _, _, L, L = esm_row_attentions.shape
        esm_row_attentions = esm_row_attentions.reshape(144, L, L).transpose(1, 2, 0)
        y_train1 = copy.deepcopy(data['Mon_distance'])
        y_train1 = np.reshape(y_train1, y_train1.shape + (1,))
        
        dock_map = data['Docking_mapcut6']
        dock_map = np.reshape(dock_map, dock_map.shape + (1,))
        pair_data = None

        if is_feature_2D[2]:
            pair_data = y_train1 if pair_data is None else np.concatenate((pair_data, y_train1), axis=2)
        if is_feature_2D[1]:
            pair_data = dock_map if pair_data is None else np.concatenate((pair_data, dock_map), axis=2)
        if is_feature_2D[0]:
            pair_data = esm_row_attentions if pair_data is None else np.concatenate((pair_data, esm_row_attentions),
                                                                                    axis=2)

        # transpose the dimension for the( B,C,L,L)
        seq_data = seq_data.transpose(1, 0).astype(np.float32)
        pair_data = pair_data.transpose(2, 0, 1).astype(np.float32)

        # Target data
        traget = data['Distance4label']
        traget[traget == 0] = np.inf

        # valid mask
        mask = np.isfinite(traget)
        data_all.append([torch.tensor(seq_data), torch.tensor(pair_data), torch.tensor(traget), torch.tensor(mask)])
    return data_all


def train_valid_test_pdb_list(pdb_path):
    train_list = [line.strip('\n') for line in open(pdb_path + "train.txt", 'r').readlines()]
    valid_list = [line.strip('\n') for line in open(pdb_path + "valid.txt", 'r').readlines()]
    test_list = [line.strip('\n') for line in open(pdb_path + "test.txt", 'r').readlines()]
    
    CASP_CAPRI_test_list = [line.strip('\n') for line in open(pdb_path + "CASP_CAPRI_test.txt", 'r').readlines()]
    test_list_218 = [line.strip('\n') for line in open(pdb_path + "test_list_218.txt", 'r').readlines()]
    
    test_list_AF2 = [line.strip('\n') for line in open(pdb_path + "test.txt", 'r').readlines()]
    CASP_CAPRI_test_list_AF2 = [line.strip('\n') for line in open(pdb_path + "CASP_CAPRI_test.txt", 'r').readlines()]
    test_list_218_AF2 = [line.strip('\n') for line in open(pdb_path + "test_list_218.txt", 'r').readlines()]
    return train_list, valid_list, test_list, test_list_218, CASP_CAPRI_test_list,test_list_AF2,test_list_218_AF2,CASP_CAPRI_test_list_AF2


def train_valid_test_pdb_file(pdb_path):
    data_path_train = os.path.join(pdb_path + "/train")
    data_path_valid = os.path.join(pdb_path + "/valid")
    data_path_test = os.path.join(pdb_path + "/test")
    data_path_test_AF2 = os.path.join(pdb_path + "/test_AF2")
    data_path_CASP_CAPRI_test = os.path.join(pdb_path + "/CASP_CAPRI_test")
    data_path_CASP_CAPRI_test_AF2 = os.path.join(pdb_path + "/CASP_CAPRI_test_AF2")
    data_path_test_list_218 = os.path.join(pdb_path + "/test_218")
    data_path_test_list_218_AF2 = os.path.join(pdb_path + "/test_218_AF2")
    return data_path_train, data_path_valid, data_path_test, data_path_test_list_218, data_path_CASP_CAPRI_test,data_path_test_AF2,data_path_test_list_218_AF2,data_path_CASP_CAPRI_test_AF2
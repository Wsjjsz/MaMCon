#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 上午9:38
# @Author  : Shujia Wei
# @File    : utils.py


import torch
from torch.autograd import Variable
import numpy as np
from dataset.dataset import read_data_from_file
import os
import matplotlib.pyplot as plt


def CUDA_selected(data):
    # The CDUA is chosen according to the situation
    if torch.cuda.is_available():
        return Variable(data.cuda())
    else:
        return Variable(data)


def TruncateProtein(data, max_length=None, random=False):
    """
        Random maximum truncation
    :param data:
    :param max_length: 400
    :param random: True
    :return:
    """
    _, length = data[0].shape
    num_residue = length
    if num_residue > max_length:
        if random:
            start = torch.randint(num_residue - max_length, (1,)).item()
        else:
            start = 0
        end = start + max_length
        mask = torch.zeros(num_residue, dtype=torch.bool)
        mask[start:end] = True
        for item in range(len(data)):
            if data[item].dim() == 2:
                i, j = data[item].shape
                if i != j:
                    data[item] = data[item][:, start:end]
                else:
                    data[item] = data[item][start:end, start:end]
            elif data[item].dim() == 3:
                data[item] = data[item][:, start:end, start:end]
        return data
    else:
        return data


def iterate_minibatches(ProteinLists, path_list_class, args, shuffle=True):
    batchsize = args.batch_size
    dim_1D = args.dim_1D
    dim_2D = args.dim_2D
    Truncate_length = args.Truncate_length
    indices = np.arange(len(ProteinLists))
    if shuffle:
        np.random.shuffle(indices)
    maxLength = 0
    inputs_1d = torch.zeros(size=(batchsize, dim_1D, Truncate_length), dtype=torch.float32)
    inputs_2d = torch.zeros(size=(batchsize, dim_2D, Truncate_length, Truncate_length), dtype=torch.float32)
    masks = torch.zeros(size=(batchsize, Truncate_length, Truncate_length), dtype=torch.bool)
    targets = torch.zeros(size=(batchsize, Truncate_length, Truncate_length), dtype=torch.float32)
    for idx in range(len(ProteinLists)):
        if idx % batchsize == 0:
            inputs_1d.fill_(0)
            inputs_2d.fill_(0)
            masks.fill_(False)
            targets.fill_(0)
            batch_idx = 0
            maxLength = 0
        data = read_data_from_file([ProteinLists[indices[idx]]], path_list_class, args)[0]
        _, length = data[0].shape
        if length > Truncate_length:
            data = TruncateProtein(data, Truncate_length, args.Truncate_random)
            length = Truncate_length

        inputs_1d[batch_idx, :, :length] = data[0]
        inputs_2d[batch_idx, :, :length, :length] = data[1]
        targets[batch_idx, :length, :length] = data[2]
        masks[batch_idx, :length, :length] = data[3]
        batch_idx += 1
        if length > maxLength:
            maxLength = length
        if (idx + 1) % batchsize == 0:
            yield inputs_1d[:, :, :maxLength], inputs_2d[:, :, :maxLength, :maxLength], targets[:, :maxLength,:maxLength], masks[:,:maxLength,:maxLength]
    if len(ProteinLists) % batchsize != 0:
        yield inputs_1d[:, :, :maxLength], inputs_2d[:, :, :maxLength, :maxLength], targets[:, :maxLength,:maxLength], masks[:, :maxLength,:maxLength]


def train(args, model, train_list, path_train_file, optimizer, criterion):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in iterate_minibatches(train_list, path_train_file, args, shuffle=True):
        inputs_1d, inputs_2d, targets, masks = batch
        input1D, input2D, target, masks = CUDA_selected(inputs_1d), CUDA_selected(inputs_2d), CUDA_selected(
            targets), CUDA_selected(masks.unsqueeze(dim=1))

        optimizer.zero_grad()
        outputs = model(input1D, input2D)
        
        target = target < args.threshold

        if args.mask == 1:
            loss = criterion(target, outputs.squeeze(), masks.squeeze(dim=0))
        else:
            loss = criterion(target, outputs)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    return total_loss / count


"""
    Deephomo2.0：
    论文中提供的评估代码
    arg.metrics_one
"""


def TopAccuracy(pred=None, truth=None, mask=None, top=["L/1", "L/2", "L/5", "L/10", 1, 10, 25, 50, 100], gap=6):
    """
    Arguments:
        pred  : predicted contact matrix
        truth : true contact matrix
        top   : evaluation for the top number
        diag  : Whether to consider the diagonal for the evaluation
    Return:
        acc   : a array of the precision of top n
    """

    if pred is None:
        print('please provide a predicted contact matrix')
        exit(1)

    if truth is None:
        print('please provide a true distance matrix')
        exit(1)

    assert pred.shape == truth.shape
    # reshape, transpose for symmetry
    L = pred.shape[-1]
    pred = pred.detach().cpu().numpy().reshape(L, L)
    truth = truth.detach().cpu().numpy().reshape(L, L)
    mask = mask.detach().cpu().numpy().reshape(L, L)
    avg_pred = pred
    pred_truth = np.dstack((avg_pred, truth))

    mask = np.triu(mask, gap)

    res = pred_truth[(mask > 0) & (truth >= 0)]
    cov = pred_truth[(mask > 0) & (truth >= 1)]
    Tn = len(cov)
    if res.size == 0:
        print("ERROR: No prediction")
        exit()
    res_sorted = res[(-res[:, 0]).argsort()]
    # calculate the precisions
    accs = {}
    for numTops in top:
        name = numTops
        numTops = str(numTops)
        if numTops[0] == "L":
            k = numTops[2:]
            numTops = round(L / int(k))
        elif numTops == "cov":
            numTops = Tn
        else:
            numTops = min(int(numTops), res_sorted.shape[0])
        if numTops == 0:
            accs[name] = 0
        else:
            topLabels = res_sorted[:numTops, 1]
            numCorrects = (topLabels == 1).sum()
            accuracy = numCorrects * 1. / numTops
            accs[name] = accuracy
    return accs

def plot_contacts_and_predictions(contacts,target, PdbID, long_range_pl, save_file_path, textId, animated=False):
    if isinstance(contacts, torch.Tensor):
        contacts = contacts.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # Create 1 row, 2 columns of subplots
    l = contacts.shape[0]
    
    # Set the contacts plot for the first subplot
    invalid_mask = np.abs(np.add.outer(np.arange(l), -np.arange(l))) < 6
    contacts[invalid_mask] = False
    title_text = PdbID + f" P@L: {100 * long_range_pl:0.1f}"
    print(invalid_mask)
    target[invalid_mask] = False
    # First subplot: contacts plot
    cax1 = ax[0].imshow(contacts, cmap="viridis", animated=animated)
    ax[0].set_title(f"{title_text}")
    ax[0].axis("square")
    ax[0].set_xlim([0, l])
    ax[0].set_ylim([l, 0])
    fig.colorbar(cax1, ax=ax[0], pad=0.1, aspect=25,shrink=0.75)
    ax[0].text(0.5, -0.1, "MaMCon", horizontalalignment='center',
               verticalalignment='bottom', transform=ax[0].transAxes,
               fontsize=18)
    # Second subplot: modified contacts plot
    cax2 = ax[1].imshow(target, cmap="viridis", animated=animated)
    ax[1].set_title(f"{title_text}")
    ax[1].axis("square")
    ax[1].set_xlim([0, l])
    ax[1].set_ylim([l, 0])
    

    # Add the textId at the bottom of the figure
    ax[1].text(0.5, -0.1, "GroundTruth", horizontalalignment='center',
               verticalalignment='bottom', transform=ax[1].transAxes,
               fontsize=18)
    fig.colorbar(cax2, ax=ax[1], pad=0.1, aspect=25,shrink=0.75)
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_file_path)


def eval(args, model, eval_list, path_eval_file, isprint=False,is_test=True):
    model.eval()
    metric = {}
    for TOP in args.metrics:
        metric[TOP] = 0.0
    total_num = 0

    _, tail = os.path.split(path_eval_file)
    statistics_path = args.statistics_path + args.log
    i = 0
    if isprint:
        if not os.path.exists(statistics_path):
            os.makedirs(statistics_path)
        if not os.path.exists(statistics_path + "/" + tail):
            os.makedirs(statistics_path + "/" + tail)
        if not os.path.exists(statistics_path + "/data_" + tail):
            os.makedirs(statistics_path + "/data_" + tail)
        open(statistics_path + "/" + tail + ".txt", 'w', encoding='utf-8')
    with torch.no_grad():
        for batch in iterate_minibatches(eval_list, path_eval_file, args, shuffle=False):
            inputs_1d, inputs_2d, targets, masks = batch
            input1D, input2D, target, masks = CUDA_selected(inputs_1d), CUDA_selected(inputs_2d), CUDA_selected(
                targets), CUDA_selected(masks.unsqueeze(dim=1))
            y_pred = model(input1D, input2D)
            target = target.squeeze()
            target = target < args.threshold
            prediction = np.array(y_pred.cpu()).squeeze()
            prediction = (prediction + prediction.transpose((1, 0))) / 2.0
            prediction = CUDA_selected(torch.tensor(prediction))
            
            if is_test:
                metric_ = TopAccuracy(prediction.squeeze(), target, masks.squeeze(), top=args.metrics, gap=args.gap)
            else:
                metric_ = TopAccuracy(prediction.squeeze(), target, masks.squeeze(), top=args.metrics, gap=args.valid_gap)
            if isprint:
                np.save(statistics_path + "/data_" + tail + "/" + eval_list[i] + ".npy", prediction.detach().cpu().numpy())
                plot_contacts_and_predictions(prediction, target,eval_list[i], metric_["L/1"], statistics_path + "/" + tail + "/" + eval_list[i] + "_pred.png", "Mamba")
                with open(statistics_path + "/" + tail + ".txt", 'a', encoding='utf-8') as file:
                    file.write(" " + str(i + 1) + "th  ID:" + eval_list[i] + "\n")
                    for key, value in metric_.items():
                        line = f"{key}: {value * 100}%\n"
                        file.write(line)
                    file.write("--------" * 10 + "\n")
                with open(statistics_path + "/" + tail + "_metric.txt", 'a', encoding='utf-8') as file1:
                    topstr = ""
                    topstr = eval_list[i]
                    for key, value in metric_.items():
                        line = f" {key}: {value * 100}% "
                        topstr += line
                    file1.write(topstr + "\n")
            i = i + 1

            total_num = total_num + 1
            for TOP in metric_.keys():
                metric[TOP] = metric_[TOP] + metric[TOP]
        for TOP in metric_.keys():
            metric[TOP] = metric[TOP] / total_num
        for i in metric:
            metric[i] = metric[i].item()
    return metric

import os
import pickle as pk
import statistics as stat
import re

import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import PILToTensor, ToPILImage
from PIL import Image


# --- LOADING UTILS ---------

def stringToIntList(string: str) -> list:
    lst = [int(s) for s in np.array(re.sub("[\[,'\]]", '', string).split(' '))]
    return np.array(lst)

def stringToStrList(string: str) -> np.array:
    return np.array(re.sub("[\[,'\]\\n]", '', string).split(' '))

def stringToFloatList(string: str) -> list:
    lst = [float(s) for s in np.array(re.sub("[\[,'\]]", '', string).split(' '))]
    return np.array(lst)

# --- MODEL UTILS ----------
def get_padding_mask(lengths):
    lengths = [i + 1 for i in lengths]
    res = []
    for i in lengths:
        res.append(torch.ones(i, device=device))
    return torch.cat([pad_sequence(res, batch_first=True) < 0.1,
                      torch.zeros(len(lengths), 1, device=device) < 1], 1)

def get_padding_mask(lengths, seq_len, device):
    lengths = [i + 1 for i in lengths]
    res = []
    for i in lengths:
        #res.append(torch.ones(i, device=device))
        if i > seq_len + 1:
            res.append(torch.ones(seq_len + 1, device=device))
        else:
            res.append(torch.cat([torch.ones(i, device=device), torch.zeros(seq_len - i + 1, device=device)]))
    return pad_sequence(res, batch_first=True) < 0.1

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def pad_seq_fn(data): # функция для выравнивания последовательностей по максимальной длине
    geo_seq, g_embs, labels, imgs_embed, extras, lenghths = zip(*data) # для варианта seq, target  
    geo_seq = pad_sequence(geo_seq, batch_first=True, padding_value=PADDING_VALUE)
    g_embs = pad_sequence(g_embs, batch_first=True, padding_value=-1)
    labels = torch.stack([torch.tensor(label) for label in labels])
    imgs_embed = pad_sequence(imgs_embed, batch_first=True, padding_value=PADDING_VALUE)
    extras = torch.tensor(extras, dtype=torch.float32)
        
    return geo_seq, torch.tensor(g_embs), labels, imgs_embed, extras, torch.tensor(lenghths)

def make_positional_encoding(max_length, embedding_size): # Позиционное кодирование для трансформера
    time = np.pi * torch.arange(0, max_length).float()
    freq_dividers = torch.arange(1, embedding_size // 2 + 1).float()
    inputs = time[:, None] / freq_dividers[None, :]
    result = torch.zeros(max_length, embedding_size)
    result[:, 0::2] = torch.sin(inputs)
    result[:, 1::2] = torch.cos(inputs)
    return result

def train(model, optimizer, device, data, graph_embed, labels, imgs_embed, extras, lengths):
    optimizer.zero_grad()
    output = model(graph_embed, imgs_embed, extras, lengths)
        
    labels = labels.to(device=device, dtype=torch.float)
    one_value = torch.tensor(0.01).to(device)
        
    loss_mape = mape_loss(labels, output, one_value)
    loss_mae = mae_loss(output, labels)
    loss_sr = sr_loss(labels, output, one_value)
    loss_rmse = torch.sqrt(mse_fn(output, labels))
    
    loss_mape.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()
    
    return loss_mape.item(), loss_mae.item(), loss_sr.item(), loss_rmse.item()


@torch.no_grad()
def test(model, device, data, graph_embed, labels, imgs_embed, extras, lengths):
    output = model(graph_embed, imgs_embed, extras, lengths)
    
    labels = labels.to(device=device, dtype=torch.float)
    one_value = torch.tensor(0.01).to(device)

    loss_mape = mape_loss(labels, output, one_value)
    loss_mae = mae_loss(output, labels)
    loss_sr = sr_loss(labels, output, one_value)
    loss_rmse = torch.sqrt(mse_fn(output, labels))

    return loss_mape.item(), loss_mae.item(), loss_sr.item(), loss_rmse.item()


criterion = torch.nn.L1Loss(reduce=False)
mae_loss = torch.nn.L1Loss()
mse_fn = nn.MSELoss()

def sr_loss(true_val, pred_val, one_value):
    return (sum(criterion(true_val, pred_val) /
                torch.where(true_val > 0.001, true_val, one_value) < 0.1) / float(true_val.shape[0])
           ) * 100

def mape_loss(true_val, pred_val, one_value):
    return torch.mean(criterion(true_val, pred_val) /
                      torch.where(true_val > 0.001, true_val, one_value))

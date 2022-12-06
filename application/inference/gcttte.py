#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import pickle as pk
import statistics as stat

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.nn.models import DeepGraphInfomax
from torch.nn.utils.rnn import pad_sequence

import re
import gc

from configs import *

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def make_positional_encoding(max_length, embedding_size): # Позиционное кодирование для трансформера
    time = np.pi * torch.arange(0, max_length).float()
    freq_dividers = torch.arange(1, embedding_size // 2 + 1).float()
    inputs = time[:, None] / freq_dividers[None, :]
#     print('input', inputs.shape)
    result = torch.zeros(max_length, embedding_size)
#     print('res', result.shape)
    result[:, 0::2] = torch.sin(inputs)
    result[:, 1::2] = torch.cos(inputs)
    return result

def get_padding_mask(lengths, seq_len):
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


class GeoConv(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(GeoConv, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.build()

    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj: torch.tensor):
        lngs = traj[:, :, 0]
        lats = traj[:, :, 1]
        states = traj[:, :, 2]
        
        lngs = torch.unsqueeze(lngs, dim = 2)
        lats = torch.unsqueeze(lats, dim = 2)
        
        states = self.state_em(states.long())

        locs = torch.cat((lngs, lats, states), dim = 2)

        # map the coords into 16-dim vector
        locs = F.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)

        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        return conv_locs


class GraphRecurrentNet(nn.Module):
    def __init__(self,
                 data,
                 graph_feat_size,
                 seq_length,
                 model_hidden_size=128,
                 n_graph_layers=3,
                 use_infomax=1,
                 graph_model_name="GCNConv",
                 num_transf_enc_layers=6,
                 n_out_layers=3,
                 alpha_g=1.0,
                 alpha_feat=0.1,
                 linear_size=32,
                ):
        super(GraphRecurrentNet, self).__init__()
        
        self.data = data
        self.seq_length = seq_length
        self.graph_feat_size = graph_feat_size
        self.model_hidden_size = model_hidden_size
        self.alpha_g = alpha_g
        self.alpha_feat = alpha_feat
        
        # joint layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_hidden_size*2 + linear_size, nhead=N_HEADS)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_transf_enc_layers)

        # graph layers
        self.n_graph_layers = n_graph_layers
        for i in range(n_graph_layers):
            if i == 0:
                print(graph_model_name)
                if graph_model_name == 'RGCNConv':
                    setattr(self,
                        'graph_{}'.format(i),
                        getattr(torch_geometric.nn, graph_model_name)(self.graph_feat_size,
                                                                      model_hidden_size,
                                                                      self.data.edge_index.shape[1])
                           )
                else:
                    setattr(self,
                            'graph_{}'.format(i),
                            getattr(torch_geometric.nn, graph_model_name)(self.graph_feat_size, model_hidden_size))
            else:
                if graph_model_name == 'RGCNConv':
                    setattr(self,
                            'graph_{}'.format(i),
                            getattr(torch_geometric.nn, graph_model_name)(model_hidden_size,
                                                                          model_hidden_size,
                                                                          self.data.edge_index.shape[1])
                           )
                else:
                    setattr(self,
                            'graph_{}'.format(i),
                            getattr(torch_geometric.nn, graph_model_name)(model_hidden_size,
                                                                          model_hidden_size)
                           )
        if use_infomax:
            if n_graph_layers == 1:
                self.info_max_layer = DeepGraphInfomax(hidden_channels=model_hidden_size, 
                                                   encoder=getattr(self,'graph_{}'.format(n_graph_layers - 1)),
                                                   summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                                                   corruption=corruption
                                                  )
            else:
                self.info_max_layer = DeepGraphInfomax(hidden_channels=model_hidden_size, 
                                                   encoder=getattr(self,'graph_{}'.format(n_graph_layers - 1)),
                                                   summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                                                   corruption=corruption
                                                  )
        self.use_infomax = use_infomax
        # make last layers
        layers = []
        for i in range(n_out_layers):
            if i == 0:
                layers.append(nn.Linear(model_hidden_size * 2 + linear_size, model_hidden_size * 2 + linear_size))
                out_size = model_hidden_size * 2 + linear_size
                layers.append(nn.ReLU())
            elif i == n_out_layers - 1:
                layers.append(nn.Linear(out_size, 1))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(out_size, out_size // 2))
                out_size = out_size // 2
                layers.append(nn.ReLU())
        self.output_layer = nn.Sequential(*layers)

        self.extras2decoder = nn.Linear(10, linear_size) # Вот эту часть точно мужно менять
        
    def forward(self, geo_seq, graph_embed, imgs_embed, extras, lengths):
        # === Обработка нужных эмбеддингов картинок ====================================
        
        imgs_embed = torch.zeros(graph_embed.shape[0], self.seq_length+1, self.model_hidden_size, device=device)
        imgs_embed += make_positional_encoding(imgs_embed.shape[1], imgs_embed.shape[2]).unsqueeze(0).to(device=device) # Делаем позиционное кодирование для трансформера
        
        
        # === Graph processing ==============================================
        
        for i in range(self.n_graph_layers):
            if i == 0:
                x = F.relu(self.graph_0(self.data.x, self.data.edge_index, None))
            else:
                x = F.relu(getattr(self, 'graph_{}'.format(i))(x, self.data.edge_index, None))
                
        if self.use_infomax:
            x, _, _ = self.info_max_layer(x, self.data.edge_index,)
        
        graph_embs = []
        for route in graph_embed:

            route = torch.stack([j for j in route if j != -1], dim=0)
            graph_embs.append(x[route])
        graph_embs = pad_sequence(graph_embs, batch_first=True, padding_value=0)
        
        if not PATH_BLIND:
            graph_embs = torch.cat([torch.zeros(graph_embs.shape[0], 1, graph_embs.shape[2]).to(device=device),
                                    graph_embs,
                                    torch.zeros(graph_embs.shape[0],
                                                self.seq_length - graph_embs.shape[1],
                                                graph_embs.shape[2]).to(device=device),
                                   ], dim=1)
        graph_embs += make_positional_encoding(graph_embs.shape[1], graph_embs.shape[2]).unsqueeze(0).to(device=device)
        
        #diffdim = imgs_embed.shape[0] - geo_seq.shape[1]
        #geo_seq = torch.cat([geo_seq, torch.zeros(geo_seq.shape[0], diffdim, geo_seq.shape[2], device=device)], dim=1)
        
        
        extras = self.extras2decoder(extras.to(device=device, dtype=torch.float)).unsqueeze(1)
#         print(graph_embs.shape, imgs_embed.shape, extras.repeat(1, imgs_embed.shape[1], 1).shape)
        # === Здесь мы пытаемся обработать получившиеся последовательности =============
        
        # print("graph_embs", graph_embs.shape) #FIXME
        # print("imgs_embed", imgs_embed.shape) #FIXME
        # print("extras", extras.shape) #FIXME
    
        input_for_TrEncoder = torch.cat([graph_embs * self.alpha_g,
                                         imgs_embed * (1.0 - self.alpha_g),
                                         self.alpha_feat * extras.repeat(1, imgs_embed.shape[1], 1),
                                        ], dim=2)
#         if PATH_BLIND:
#             pad_mask = get_padding_mask(lengths=2*torch.ones(1, graph_embs.shape[0]).to(device))
#         else:
        if not PATH_BLIND:
            pad_mask = get_padding_mask(lengths=lengths, seq_len=self.seq_length) 
            # print(pad_mask.shape, input_for_TrEncoder.transpose(0, 1).shape, lengths) #FIXME
            out_transformer = self.transformer_encoder(input_for_TrEncoder.transpose(0, 1),
                                                   src_key_padding_mask=pad_mask)
        else:
            out_transformer = self.transformer_encoder(input_for_TrEncoder.transpose(0, 1)) # Энкодер для графа
        out_transformer = out_transformer.mean(0)
    
        return self.output_layer(out_transformer)
    
    
class GCTTTE:
    def __init__(self,
                 model,
                 sequence_length=128):
        self.sequence_length=sequence_length
        self.model=model
        
    def predict(self, route, extra_features):
        # route_2_nodeIdSeq[index] можно заменить на любую последовательность индексов np.array()
        graph_emb_ids = torch.tensor(route, dtype=torch.long, device=device)[:self.sequence_length]
        if graph_emb_ids.shape[0] < self.sequence_length:
            minus_ones = torch.zeros(self.sequence_length - graph_emb_ids.shape[0], dtype=torch.long, device=device) - 1
            graph_emb_ids = torch.cat([graph_emb_ids, minus_ones])
        graph_emb_ids = graph_emb_ids.unsqueeze(0)
 
        route2Tids = torch.zeros(1,  self.sequence_length+1, dtype=torch.int)
            
        leng = len(route) + 1
        
        return self.model(None,
                          graph_emb_ids,
                          route2Tids,
                          torch.tensor(extra_features).unsqueeze(0),
                          torch.tensor(leng).unsqueeze(0))
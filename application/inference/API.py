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
from sklearn.preprocessing import StandardScaler

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.nn.models import DeepGraphInfomax

import re
import gc

from gcttte import GCTTTE as API
from gcttte import GraphRecurrentNet

from configs import *

print("Getting x data")
x_Abakan = pd.read_csv(X_ABAKAN_PATH).to_numpy()

print("Getting edge_index data")
edge_index_Abakan = np.load(open(EDGE_INDEX_ABAKAN, 'rb')) # Возможно, он тут не нужен

print("Getting x data")
x_Omsk = pd.read_csv(X_OMSK_PATH).to_numpy()

print("Getting edge_index data")
edge_index_Omsk = np.load(open(EDGE_INDEX_OMSK, 'rb')) # Возможно, он тут не нужен

class GCT_TTE_inference():
    
    _instance = None

    def __new__(cls): # make it singleton
        if cls._instance is None:
            cls._instance = super(GCT_TTE_inference, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.dataA = Data(
            x=torch.tensor(x_Abakan, dtype=torch.float), # признаковое описание
            edge_index=torch.tensor(edge_index_Abakan, dtype=torch.long), # транспонированный список ребер
        ).to(device)
        
        self.dataO = Data(
            x=torch.tensor(x_Omsk, dtype=torch.float), # признаковое описание
            edge_index=torch.tensor(edge_index_Omsk, dtype=torch.long), # транспонированный список ребер
        ).to(device)
        
        self.modelA = GraphRecurrentNet(self.dataA,
                      graph_feat_size=GRAPH_FEAT_SIZE_ABAKAN,
                      seq_length=SEQ_LEN_ABAKAN,
                      model_hidden_size=HIDDEN_SIZE,
                      n_graph_layers=n_graph_layers,
                      use_infomax=use_infomax,
                      graph_model_name=graph_model,
                      num_transf_enc_layers=n_encoder_layers,
                      n_out_layers=n_fuse_layers,
                      alpha_g=alpha_g,
                      alpha_feat=alpha_feat,
                      linear_size=LINEAR_SIZE,
                     ).to(device)
        self.modelA.load_state_dict(torch.load(ABAKAN_STATE_DICT))
        
        self.apiA = API(self.modelA)
        
        self.modelA.to(device)
        self.modelA.eval()
        
        self.modelO = GraphRecurrentNet(self.dataO,
                      graph_feat_size=GRAPH_FEAT_SIZE_OMSK,
                      seq_length=SEQ_LEN_OMSK,
                      model_hidden_size=HIDDEN_SIZE,
                      n_graph_layers=n_graph_layers,
                      use_infomax=use_infomax,
                      graph_model_name=graph_model,
                      num_transf_enc_layers=n_encoder_layers,
                      n_out_layers=n_fuse_layers,
                      alpha_g=alpha_g,
                      alpha_feat=alpha_feat,
                      linear_size=LINEAR_SIZE,
                     ).to(device)
        self.modelO.load_state_dict(torch.load(OMSK_STATE_DICT))
        
        self.apiO = API(self.modelO, sequence_length=SEQ_LEN_OMSK)
        
        self.modelO.to(device)
        self.modelO.eval()
        
    def predict(self, city: str, edge_ids: list, extras = EXTRAS_ABAKAN.squeeze(0)):
        edge_ids = torch.tensor(edge_ids, device=device)
        with torch.no_grad():
            if city == "Abakan":
                #graph_embed, extras, lengths
                return self.apiA.predict(edge_ids, extras).item()
            if city == "Omsk":
                return self.apiO.predict(edge_ids, extras).item()
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    
    def __init__(self,
        route_2_nodeIdSeq: pd.Series,
        targets: np.array,
        iter_2_id: np.array,
        route_2_Tids: pd.Series,
        extra_features: np.array,
        sequence_length: int,
        path_blind: bool,
    ):
        self.route_2_nodeIdSeq = route_2_nodeIdSeq
        self.targets = targets
        self.iter_2_id = iter_2_id
        self.route_2_Tids = route_2_Tids
        self.extra_features = extra_features
        self.sequence_length = sequence_length
        self.path_blind = path_blind
        
    def __getitem__(self, index):
        index = self.iter_2_id[index]
        global TENSORS, GEO_SEQ
        
        if self.path_blind:
            return (torch.tensor(self.route_2_nodeIdSeq[index])[[0, -1]], # for graph
                    torch.tensor(self.targets[index]),      
                    torch.tensor(self.route_2_Tids[index])[[0, -1]], # for patch
                    self.extra_features[index],
                    torch.tensor(max(len(self.route_2_nodeIdSeq[index]), 2)),
                   )
        
        leng = len(self.route_2_Tids[index])
        
        graph_emb_ids = torch.tensor(self.route_2_nodeIdSeq[index], dtype=torch.long)[:self.sequence_length]
        if graph_emb_ids.shape[0] < self.sequence_length:
            minus_ones = torch.zeros(self.sequence_length - graph_emb_ids.shape[0], dtype=torch.long) - 1
            graph_emb_ids = torch.cat([graph_emb_ids, minus_ones])
        
        route2Tids = torch.tensor(self.route_2_Tids[index], dtype=torch.int)[:self.sequence_length+1]
        if route2Tids.shape[0] < self.sequence_length+1:
            route2Tids = torch.cat([route2Tids, torch.zeros(self.sequence_length+1 - route2Tids.shape[0], dtype=torch.int)])
        
        return (graph_emb_ids, # for graph
                torch.tensor(self.targets[index]),      
                route2Tids, # for patch
                self.extra_features[index],
                torch.tensor(leng),
               )
    
    def __len__(self):
        return len(self.iter_2_id)

import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import networkx as nx


class LaplacianEmbedding(nn.Module):
    def __init__(self, vertexes_number, embedding_size=20):
        super(LaplacianEmbedding, self).__init__()
        self.vertexes_number, self.embedding_size = vertexes_number, embedding_size
        self.weights = torch.Tensor(self.vertexes_number, self.embedding_size)
        self.weights = nn.Parameter(self.weights)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, A):
        result = torch.Tensor([0]).to(device)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                result += A[i][j] * torch.norm(self.weights[i] - self.weights[j])
        return result, self.weights


class ResLinearBlock(nn.Module):
    def __init__(self):
        super(ResLinearBlock, self).__init__()
        self.linear_1 = nn.Linear(1024, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.batch_norm_1 = nn.BatchNorm1d(1024)
        self.batch_norm_2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        result = self.linear_1(x)
        result = F.relu(self.batch_norm_1(result))
        result = self.linear_2(result)
        result = F.relu(self.batch_norm_2(result))

        return x + result


class MURAT(nn.Module):
    def __init__(self):
        super(MURAT, self).__init__()

        # Spatial and temporal graph embeddings
        self.spatial_emb = LaplacianEmbedding(30 * 30)
        self.temporal_emb = LaplacianEmbedding(31)

        # Convert to residual network input
        self.linear_0 = nn.Linear(20 + 20 + 40, 1024)

        # Residual network
        self.res_block_1 = ResLinearBlock()
        self.res_block_2 = ResLinearBlock()
        self.res_block_3 = ResLinearBlock()
        self.res_block_4 = ResLinearBlock()

        # Final part
        self.linear_1 = nn.Linear(1024, 1)

    def forward(
        self,
        link_embeddings,
        spatial_adj_matrix,
        spatial_indices,
        temporal_adj_matrix,
        temporal_indices,
    ):
        summ_s, spatial = self.spatial_emb(spatial_adj_matrix)
        summ_t, temporal = self.temporal_emb(temporal_adj_matrix)

        spatial_matrix = [
            spatial[spatial_indices[i][0] * 30 + spatial_indices[i][1]]
            for i in range(len(spatial_indices))
        ]
        temporal_matrix = [
            temporal[temporal_indices[i][0] * 7 + temporal_indices[i][1]]
            for i in range(len(temporal_indices))
        ]

        spatial_tensor = torch.stack(spatial_matrix, dim=0)
        temporal_tensor = torch.stack(temporal_matrix, dim=0)

        resnet_input = torch.cat(
            [link_embeddings, spatial_tensor, temporal_tensor], dim=1
        )
        resnet_input = self.linear_0(resnet_input)

        resnet_output = self.res_block_1(resnet_input)
        resnet_output = self.res_block_2(resnet_output)
        resnet_output = self.res_block_3(resnet_output)
        resnet_output = self.res_block_4(resnet_output)

        final_output = F.relu(self.linear_1(resnet_output))
        return final_output, summ_s, summ_t
    

if __name__ == "__main__":
    model = MURAT()
 
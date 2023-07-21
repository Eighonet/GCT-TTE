import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.models import DeepGraphInfomax
import torch_geometric

from utils import corruption, make_positional_encoding, get_padding_mask


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


class GCTTTE(nn.Module):
    
    def __init__(self,
                 data,
                 model_hidden_size=128,
                 n_graph_layers=3,
                 use_infomax=1,
                 graph_model_name="GCNConv",
                 num_transf_enc_layers=6,
                 n_out_layers=3,
                 alpha_g=1.0,
                 alpha_feat=0.1,
                 linear_size=32,
                 n_heads=4,
                 img_tensors=None,
                 graph_input_size=73,
                 device="cpu",
                 path_plind=False,
                 seq_len=128,
                ):
        super(GCTTTE, self).__init__()
        
        self.data = data
        self.model_hidden_size = model_hidden_size
        self.alpha_g = alpha_g
        self.alpha_feat = alpha_feat
        self.device = device
        self.path_plind = path_plind
        self.seq_len = seq_len
        
        # === joint layers ==============================
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_hidden_size*2 + linear_size, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_transf_enc_layers)
        # === img layers ==============================
        
        self.img_embeddings = torch.nn.Embedding.from_pretrained(
            torch.cat([torch.zeros(1, img_tensors.shape[1]), img_tensors]),
            padding_idx=0)
        self.imgemb2tr1 = nn.Linear(3712, 1856)
        self.imgemb2tr2 = nn.Linear(1856, model_hidden_size)
        
        # === GeoConv layers ==============================================
        # this layers are not utilized in this version 
        # you need to rewrite forward to use them
        
        self.geo_conv = GeoConv(3, linear_size) 
        
        # === graph layers ==============================
        
        self.n_graph_layers = n_graph_layers
        for i in range(n_graph_layers):
            if i == 0:
                if graph_model_name == 'RGCNConv':
                    setattr(self,
                        'graph_{}'.format(i),
                        getattr(torch_geometric.nn, graph_model_name)(graph_input_size,
                                                                      model_hidden_size,
                                                                      self.data.edge_index.shape[1])
                           )
                else:
                    setattr(self,
                            'graph_{}'.format(i),
                            getattr(torch_geometric.nn, graph_model_name)(graph_input_size, model_hidden_size))
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
        
        # === make last layers ==============================
        
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

        self.extras2decoder = nn.Linear(10, linear_size) 
        
        
    def forward(self, graph_embed, imgs_embed, extras, lengths):
        
        # === Images processing ==============================================
        
        imgs_embed = imgs_embed.to(self.device) # to gpu

        imgs_reduced_embs = [] # stack it here
        for route in imgs_embed: # For every road in batch
            route = self.img_embeddings(route)
            img_emb = self.imgemb2tr1(route) # Reducing the dimension
            img_emb = self.imgemb2tr2(img_emb) # Reducing the dimension
            imgs_reduced_embs.append(img_emb.unsqueeze(0))
        
        imgs_embed = torch.cat(imgs_reduced_embs) # we get a complete tensor with embeddings of all images for each road in the batch
        imgs_embed += make_positional_encoding(imgs_embed.shape[1],
                                               imgs_embed.shape[2]
                                              ).unsqueeze(0).to(device=self.device) # do positional coding for the transformer
        
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
        
        if not self.path_plind:
            graph_embs = torch.cat([torch.zeros(graph_embs.shape[0], 1, graph_embs.shape[2]).to(device=self.device),
                                    graph_embs,
                                    torch.zeros(graph_embs.shape[0],
                                                self.seq_len - graph_embs.shape[1],
                                                graph_embs.shape[2]).to(device=self.device),
                                   ], dim=1)
        graph_embs += make_positional_encoding(graph_embs.shape[1], graph_embs.shape[2]).unsqueeze(0).to(device=self.device)
        
        extras = self.extras2decoder(extras.to(device=self.device, dtype=torch.float)).unsqueeze(1)
        
        # === Here we are trying to process the resulting sequences =============
        
        input_for_TrEncoder = torch.cat([graph_embs * self.alpha_g,
                                         imgs_embed * (1.0 - self.alpha_g),
                                         self.alpha_feat * extras.repeat(1, imgs_embed.shape[1], 1),
                                        ], dim=2)
        if not self.path_plind:
            pad_mask = get_padding_mask(lengths=lengths,
                                        seq_len=self.seq_len, 
                                        device=self.device) 
            out_transformer = self.transformer_encoder(input_for_TrEncoder.transpose(0, 1),
                                                       src_key_padding_mask=pad_mask)
        else:
            out_transformer = self.transformer_encoder(input_for_TrEncoder.transpose(0, 1)) # Encoder for graph
        out_transformer = out_transformer.mean(0)
    
        return self.output_layer(out_transformer)

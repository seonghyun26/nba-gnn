import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

import torch.nn as nn
"""
=== Description of the VOCSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 14]
Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

VOC_node_input_dim = 14
# VOC_edge_input_dim = 1 or 2; defined in class VOCEdgeEncoder

class LineGraphVOCNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.node_encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        self.edge_encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)

    def forward(self, batch):
        edge_encoded_features = self.edge_encoder(batch.x[:, 1].unsqueeze(1))
        node_encoded_features1 = self.node_encoder(batch.x[:, 1:15])
        node_encoded_features2 = self.node_encoder(batch.x[:, 15:29])
        
        # batch.x = self.node_encoder(batch.x)
        batch.x = edge_encoded_features + node_encoded_features1 - node_encoded_features2

        return batch

register_node_encoder('VOCNodeLG', LineGraphVOCNodeEncoder)


class LineGraphVOCEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.node_encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        self.edge_encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)

    def forward(self, batch):
        node_encoded_features = self.node_encoder(batch.edge_attr[:, :14])
        edge_encoded_features1 = self.edge_encoder(batch.edge_attr[:, 14].unsqueeze(1))
        edge_encoded_features2 = self.edge_encoder(batch.edge_attr[:, 15].unsqueeze(1))
        
        # batch.edge_attr = self.edge_encoder(batch.edge_attr)
        batch.edge_attr = node_encoded_features + edge_encoded_features1 - edge_encoded_features2

        return batch

register_edge_encoder('VOCEdgeLG', LineGraphVOCEdgeEncoder)


class LineGraphVOCNodeLapEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        dim_pe = 16
        VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.node_encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim - dim_pe)
        self.edge_encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim - dim_pe)
        
        # NOTE: LapPE
        self.linear_A = nn.Linear(2, dim_pe)
        # pe_encoder
        layers = []
        n_layers = 4
        self.linear_A = nn.Linear(2, 2 * dim_pe)
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(2 * dim_pe, dim_pe))
        layers.append(nn.ReLU())
        self.pe_encoder = nn.Sequential(*layers)

    def lapPE(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs
        
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        pos_enc = self.pe_encoder(pos_enc)
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),0. )
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe
        
        return pos_enc
    
    def forward(self, batch):
        edge_encoded_features = self.edge_encoder(batch.x[:, 1].unsqueeze(1))
        node_encoded_features1 = self.node_encoder(batch.x[:, 1:15])
        node_encoded_features2 = self.node_encoder(batch.x[:, 15:29])
        
        # batch.x = self.node_encoder(batch.x)
        batch.x = edge_encoded_features + node_encoded_features1 - node_encoded_features2
        
        # NOTE: LapPE
        pos_enc = self.lapPE(batch)
        batch.x = torch.cat((batch.x, pos_enc), 1)
        batch.pe_LapPE = pos_enc

        return batch

register_node_encoder('VOCNodeLG+LapPE', LineGraphVOCNodeLapEncoder)


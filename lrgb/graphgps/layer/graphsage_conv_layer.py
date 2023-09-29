import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer

from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register


class SAGEConvLayer(nn.Module):
    """
    GraphSAGE Layer 
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.residual = residual
        self.dropout = dropout 
        
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.BatchNorm1d(dim_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(self.dropout),
        )
        self.model = pyg_nn.SAGEConv(dim_in, dim_out)
        
    def forward(self, batch):
        x_in = batch.x
        
        batch.x = self.model(batch.x, batch.edge_index)
        batch.x = self.act(batch.x)
        
        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
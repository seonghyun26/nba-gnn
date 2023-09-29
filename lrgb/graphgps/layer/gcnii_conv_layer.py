import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer

class GCN2ConvLayer(nn.Module):
    """GCNII Layer from https://arxiv.org/abs/2007.02133.
    """
    def __init__(self, dim_in, dim_out, dropout, residual, lgvariant=0):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.model = pyg_nn.GCN2Conv(self.dim_in, alpha=0.2)
        # alpha value is set using results from the GCNII paper
        self.batchNorm = nn.BatchNorm1d(dim_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.x0, batch.edge_index)
        
        batch.x = self.batchNorm(batch.x)
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
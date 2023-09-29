from torch import nn
import torch
from tqdm import tqdm 
from torch_scatter import segment_coo, scatter
from model.modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule, GatedGraphConvModule)

MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule],
    'gated_GCN':[GatedGraphConvModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, n_steps):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout, 
                                                        n_steps=n_steps)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, dataset):
        graph = dataset.graph
        x = dataset.node_features 
        nbtm = dataset.nbtm 
        node_edges = dataset.node_edges

        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x)

        if nbtm: 
            x =x.unsqueeze(0)
            index = node_edges.to(dtype=torch.int64)
            x = scatter(x, index, dim = 1, reduce="mean")
            x = x.squeeze(0)
        return x

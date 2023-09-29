import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.mlp_layer import MLPLayer
from graphgps.layer.gcn_layer import GCNConvLayer
from graphgps.layer.lgnn_layer import graph2linegraph, linegraph2graph, lg2graphNode, pcqmPostProcess

from torch_scatter import scatter
from torch_geometric.nn import JumpingKnowledge

class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # if cfg.gnn.lgvariant == 30:
        #     self.g2lg = g2lg()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        self.model_type = cfg.gnn.layer_type
        layers = []
            
        if cfg.gnn.linegraph:
            if cfg.gnn.lgvariant >= 12:
                print("FLAG - LG dataset2")
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in,
                        dim_in,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
            else:
                assert("Deprecated version of linegraph")
        else:
            for _ in range(cfg.gnn.layers_mp):
                layers.append(conv_model(
                    dim_in,
                    dim_in,
                    dropout=cfg.gnn.dropout,
                    residual=cfg.gnn.residual,
                ))
        
        if cfg.gnn.linegraph and cfg.gnn.lgvariant >= 20 and cfg.gnn.lgvariant <= 29:
            layers.append(lg2graphNode())
        elif cfg.gnn.linegraph and cfg.gnn.lgvariant >= 30 and cfg.gnn.lgvariant <= 39:
            layers.append(pcqmPostProcess())
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

            

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcniiconv':
            return GCN2ConvLayer
        elif model_type == 'gatconv':
            return GATConvLayer
        elif model_type == 'gatv2conv':
            return GATv2ConvLayer
        elif model_type == 'mlp':
            return MLPLayer
        elif model_type == "gcnconv":
            return GCNConvLayer
        elif model_type == "sageconv":
            return SAGEConvLayer
        elif model_type == "resgatedconv":
            return ResGatedConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for idx, module in enumerate(self.children()):
            if self.model_type == 'gcniiconv':
                batch.x0 = batch.x # gcniiconv needs x0 for each layer
                batch = module(batch)
            else:
                batch = module(batch)
                # if cfg.gnn.lgvariant == 13 and idx == 1 :
                #     xJump = [batch.x]
                #     for layerIdx, layer in enumerate(module):
                #         batch = layer(batch)
                #         xJump.append(batch.x)
                #     # batch.x = self.jump(xJump)
                #     batch.x = torch.cat(xJump, dim=1)
                # elif idx != 3:
                #     batch = module(batch)
        return batch

register_network('custom_gnn', CustomGNN)

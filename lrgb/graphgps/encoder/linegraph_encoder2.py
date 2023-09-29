import torch
import torch.nn as nn
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)
from ogb.utils.features import get_bond_feature_dims
from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder

from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.composed_encoders import concat_node_encoders

from torch_geometric.graphgym.config import cfg

class LineGraphLapPENodeNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()
        self.version = cfg.posenc_LapPE.version
        self.expand_x = cfg.posenc_LapPE.enable
        
        dim_pe = cfg.posenc_LapPE.dim_pe

        if self.version == 3:
            assert( emb_dim % 2 == 0)
            self.emb_dim = emb_dim // 2
        elif self.version == 4:
            self.emb_dim = emb_dim 
        else:
            assert("Unsupported version for LineGraphLapPENodeNodeEncoder")
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        
        # NOTE: LapPE
        self.linear_x = nn.Linear(emb_dim, emb_dim - dim_pe)
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
        node_encoded_features1 = 0
        node_encoded_features2 = 0
        
        for AtomIdx in range(0, 9):
            node_encoded_features1 += self.atom_embedding_list[AtomIdx](batch.x[:, AtomIdx])
        for AtomIdx in range(9, 18):
            node_encoded_features2 += self.atom_embedding_list[AtomIdx-9](batch.x[:, AtomIdx])
        
        if self.version == 3:
            batch.x = torch.cat([node_encoded_features1, node_encoded_features2], dim=1)
        elif self.version == 4:
            batch.x = (node_encoded_features1 + node_encoded_features2) / 2
    
        # NOTE: LapPE
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        pos_enc = self.lapPE(batch)
        batch.x = torch.cat((h, pos_enc), 1)
        
        batch.pe_LapPE = pos_enc
        
        return batch

register_node_encoder("AtomLGNode+LapPE", LineGraphLapPENodeNodeEncoder)



class LineGraphEdgeNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()
        self.version = cfg.posenc_LapPE.version
        
        if self.version == 3:
            assert( emb_dim % 3 == 0 )
            self.emb_dim = emb_dim // 3
        elif self.version == 4 :
            self.emb_dim = emb_dim
        else:
            assert("Unsupported version for LineGraphLapPENodeNodeEncoder")
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        
    
    def forward(self, batch):
        node_encoded_features1 = 0
        node_encoded_features2 = 0
        node_encoded_features3 = 0
        
        for AtomIdx in range(9):
            node_encoded_features1 += self.atom_embedding_list[AtomIdx](batch.edge_attr[:, AtomIdx])
        for AtomIdx in range(9, 18):
            node_encoded_features2 += self.atom_embedding_list[AtomIdx-9](batch.edge_attr[:, AtomIdx])
        for AtomIdx in range(18, 27):
            node_encoded_features3 += self.atom_embedding_list[AtomIdx-18](batch.edge_attr[:, AtomIdx])

        if self.version == 3:
            batch.edge_attr = torch.cat([node_encoded_features1, node_encoded_features2, node_encoded_features3], dim=1)
        elif self.version == 4:
            batch.edge_attr = (node_encoded_features1 + node_encoded_features2 + node_encoded_features3) / 3
            
        
        return batch

register_edge_encoder('BondLGNode', LineGraphEdgeNodeEncoder)



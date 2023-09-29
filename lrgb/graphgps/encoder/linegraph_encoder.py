import torch
import torch.nn as nn
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)
from ogb.utils.features import get_bond_feature_dims
from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder

from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.composed_encoders import concat_node_encoders

from torch_geometric.graphgym.config import cfg

class LineGraphNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        edge_encoded_features = 0
        node_encoded_features1 = 0
        node_encoded_features2 = 0

        for EdgeIdx in range(3):
            edge_encoded_features += self.bond_embedding_list[EdgeIdx](batch.x[:, EdgeIdx])
        for AtomIdx in range(3, 12):
            node_encoded_features1 += self.atom_embedding_list[AtomIdx-3](batch.x[:, AtomIdx])
        for AtomIdx in range(12, 21):
            node_encoded_features2 += self.atom_embedding_list[AtomIdx-12](batch.x[:, AtomIdx])
        
        # edge_encoded_features = edge_encoded_features.repeat(1,2)
        # node_encoded_features = torch.cat([node_encoded_features1, node_encoded_features2], dim=1)
        # batch.x = edge_encoded_features + node_encoded_features
        batch.x = edge_encoded_features + node_encoded_features1 - node_encoded_features2
        
        return batch

register_node_encoder('AtomLG', LineGraphNodeEncoder)


class LineGraphLapPENodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        self.version = cfg.posenc_LapPE.version
        self.expand_x = cfg.posenc_LapPE.enable
        
        dim_pe = cfg.posenc_LapPE.dim_pe
        
        if not self.expand_x:
            emb_dim = emb_dim - dim_pe

        if self.version == 2:
            assert( emb_dim % 3 == 0)
            self.emb_dim = emb_dim // 3
        elif self.version == 3:
            assert( emb_dim % 2 == 0)
            self.emb_dim = emb_dim // 2
        elif self.version == 4:
            self.emb_dim = emb_dim 
        elif self.version == 5:
            self.emb_dim = emb_dim 
            self.linear_compress = nn.Linear(emb_dim * 2, emb_dim)
        elif self.version == 6:
            self.emb_dim = emb_dim 
            self.linear_compress = nn.Linear(emb_dim * 3, emb_dim)
        else:
            assert( emb_dim % 2 == 0)
            self.emb_dim = emb_dim
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
        
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
        # self.linear_last = nn.Linear(emb_dim, emb_dim)
        
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
        edge_encoded_features = 0
        node_encoded_features1 = 0
        node_encoded_features2 = 0
        
        
        for EdgeIdx in range(3):
            edge_encoded_features += self.bond_embedding_list[EdgeIdx](batch.x[:, EdgeIdx])
        for AtomIdx in range(3, 12):
            node_encoded_features1 += self.atom_embedding_list[AtomIdx-3](batch.x[:, AtomIdx])
        for AtomIdx in range(12, 21):
            node_encoded_features2 += self.atom_embedding_list[AtomIdx-12](batch.x[:, AtomIdx])
        
        if self.version == 1:
            batch.x = edge_encoded_features - node_encoded_features1 + node_encoded_features2
        elif self.version == 2:
            batch.x = torch.cat([edge_encoded_features, node_encoded_features1, node_encoded_features2], dim=1)
        elif self.version == 3:
            batch.x = torch.cat([node_encoded_features1, node_encoded_features2], dim=1)
        elif self.version == 4:
            batch.x = (node_encoded_features1 + node_encoded_features2) / 2
        elif self.version == 5:
            batch.x = self.linear_compress(torch.cat([node_encoded_features1, node_encoded_features2], dim=1))
        elif self.version == 6:
            batch.x = self.linear_compress(torch.cat([edge_encoded_features, node_encoded_features1, node_encoded_features2], dim=1))
        else:
            batch.x = torch.cat([node_encoded_features1, node_encoded_features2], dim=1)
    
            
        # NOTE: LapPE
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        pos_enc = self.lapPE(batch)
        batch.x = torch.cat((h, pos_enc), 1)
        
        # NOTE: preprocess
        # batch.x = self.linear_last(batch.x)
        
        batch.pe_LapPE = pos_enc
        
        return batch

register_node_encoder("AtomLG+LapPE", LineGraphLapPENodeEncoder)



class LineGraphRWSENodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        self.version = cfg.posenc_LapPE.version
        self.expand_x = cfg.posenc_RWSE.enable
        self.kernel_type="RWSE"
        dim_pe = cfg.posenc_LapPE.dim_pe
        
        if not self.expand_x:
            emb_dim = emb_dim - dim_pe

        if self.version == 2:
            assert( emb_dim % 3 == 0)
            self.emb_dim = emb_dim // 3
        elif self.version == 3:
            assert( emb_dim % 2 == 0)
            self.emb_dim = emb_dim // 2
        elif self.version == 4:
            self.emb_dim = emb_dim 
        elif self.version == 5:
            self.emb_dim = emb_dim 
            self.linear_compress = nn.Linear(emb_dim * 2, emb_dim)
        elif self.version == 6:
            self.emb_dim = emb_dim 
            self.linear_compress = nn.Linear(emb_dim * 3, emb_dim)
        else:
            assert( emb_dim % 2 == 0)
            self.emb_dim = emb_dim
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
        
        # NOTE: RWSE
        pecfg = getattr(cfg, f"posenc_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = len(pecfg.kernel.times)
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if emb_dim - dim_pe < 1:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {emb_dim}.")

        if self.expand_x:
            self.linear_x = nn.Linear(emb_dim, emb_dim - dim_pe)

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        if model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")
        
    def RWSE(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")
        
        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe
        
        return pos_enc
    
    def forward(self, batch):
        edge_encoded_features = 0
        node_encoded_features1 = 0
        node_encoded_features2 = 0
        
        
        for EdgeIdx in range(3):
            edge_encoded_features += self.bond_embedding_list[EdgeIdx](batch.x[:, EdgeIdx])
        for AtomIdx in range(3, 12):
            node_encoded_features1 += self.atom_embedding_list[AtomIdx-3](batch.x[:, AtomIdx])
        for AtomIdx in range(12, 21):
            node_encoded_features2 += self.atom_embedding_list[AtomIdx-12](batch.x[:, AtomIdx])
        
        if self.version == 1:
            batch.x = edge_encoded_features - node_encoded_features1 + node_encoded_features2
        elif self.version == 2:
            batch.x = torch.cat([node_encoded_features1, node_encoded_features2, edge_encoded_features], dim=1)
        elif self.version == 3:
            batch.x = torch.cat([node_encoded_features1, node_encoded_features2], dim=1)
        elif self.version == 4:
            batch.x = (node_encoded_features1 + node_encoded_features2) / 2
        elif self.version == 5:
            batch.x = self.linear_compress(torch.cat([node_encoded_features1, node_encoded_features2], dim=1))
        elif self.version == 6:
            batch.x = self.linear_compress(torch.cat([edge_encoded_features, node_encoded_features1, node_encoded_features2], dim=1))
        else:
            batch.x = torch.cat([node_encoded_features1, node_encoded_features2], dim=1)
    

        # NOTE: RWSE
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        pos_enc = self.RWSE(batch)
        batch.x = torch.cat((h, pos_enc), 1)
        batch.pe_RWSE = pos_enc
        
        return batch

register_node_encoder("AtomLG+RWSE", LineGraphRWSENodeEncoder)

class LineGraphEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        self.version = cfg.posenc_LapPE.version
        
        if self.version == 2:
            assert( emb_dim % 3 == 0)
            self.emb_dim = emb_dim // 3
        elif self.version == 3 or self.version == 4 or self.version == 5 or self.version == 7:
            self.emb_dim = emb_dim
        elif self.version == 6:
            self.emb_dim = emb_dim 
            self.linear_compress = nn.Linear(emb_dim * 3, emb_dim)
        else:
            self.emb_dim = emb_dim
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, self.emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
        
    
    def forward(self, batch):
        node_encoded_features = 0
        edge_encoded_features1 = 0
        edge_encoded_features2 = 0
        for AtomIdx in range(9):
            node_encoded_features += self.atom_embedding_list[AtomIdx](batch.edge_attr[:, AtomIdx])
        for edgeIdx in range(9, 12):
            edge_encoded_features1 += self.bond_embedding_list[edgeIdx-9](batch.edge_attr[:, edgeIdx])
        for edgeIdx in range(12, 15):
            edge_encoded_features2 += self.bond_embedding_list[edgeIdx-12](batch.edge_attr[:, edgeIdx])

        if self.version == 1:
            batch.edge_attr = node_encoded_features - edge_encoded_features1 + edge_encoded_features2
        elif self.version == 2:
            batch.edge_attr = torch.cat([node_encoded_features, edge_encoded_features1, edge_encoded_features2], dim=1)
        elif self.version == 3 or self.version == 4 or self.version == 5:
            batch.edge_attr = node_encoded_features
        elif self.version == 6:
            batch.edge_attr = self.linear_compress(torch.cat([node_encoded_features, edge_encoded_features1, edge_encoded_features2], dim=1))
        else:
            batch.edge_attr = node_encoded_features
            
        
        return batch

register_edge_encoder('BondLG', LineGraphEdgeEncoder)




class LineGraphMagLapPENodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        dim_pe = 16    

        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
        
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
        edge_encoded_features = 0
        node_encoded_features1 = 0
        node_encoded_features2 = 0
        
        
        for EdgeIdx in range(3):
            edge_encoded_features += self.bond_embedding_list[EdgeIdx](batch.x[:, EdgeIdx])
        for AtomIdx in range(3, 12):
            node_encoded_features1 += self.atom_embedding_list[AtomIdx-3](batch.x[:, AtomIdx])
        for AtomIdx in range(12, 21):
            node_encoded_features2 += self.atom_embedding_list[AtomIdx-12](batch.x[:, AtomIdx])
            
        batch.x = edge_encoded_features + node_encoded_features1 - node_encoded_features2
        
        # NOTE: LapPE
        pos_enc = self.lapPE(batch)
        h = self.linear_x(batch.x)

        batch.x = torch.cat((h, pos_enc), 1)
        batch.pe_LapPE = pos_enc
        
        return batch

register_node_encoder("AtomLG+MagLapPE", LineGraphMagLapPENodeEncoder)


# ds_encs = {'AtomLG':LineGraphNodeEncoder}
# pe_encs = {'LapPE': LapPENodeEncoder}

# Concat dataset-specific and PE encoders.
# for ds_enc_name, ds_enc_cls in ds_encs.items():
#     for pe_enc_name, pe_enc_cls in pe_encs.items():
#         register_node_encoder(
#             f"{ds_enc_name}+{pe_enc_name}",
#             concat_node_encoders([ds_enc_cls, pe_enc_cls],
#                                  [pe_enc_name])
#         )
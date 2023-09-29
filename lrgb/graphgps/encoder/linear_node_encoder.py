import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(cfg.share.dim_in, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch

register_node_encoder('LinearNode', LinearNodeEncoder)

class LinearNodeLGEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        assert( emb_dim % 6 == 0 )
        self.encoder1 = torch.nn.Linear(cfg.share.dim_in, emb_dim // 3)
        self.encoder2 = torch.nn.Linear(cfg.share.dim_in, emb_dim // 3)
        self.encoder3 = torch.nn.Linear(1, emb_dim // 3)

    def forward(self, batch):
        edge_encoded = self.encoder1(batch.x[:, :1])
        node_encoded2 = self.encoder2(batch.x[:, 1:6])
        node_encoded2 = self.encoder2(batch.x[:, 6:])
        batch.x = torch.cat([edge_encoded, node_encoded_features1, node_encoded_features2], dim=1)
        
        return batch

register_node_encoder('LinearNodeLG', LinearNodeLGEncoder)

class LinearNodeLGLapEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        assert( emb_dim % 6 == 0 )
        self.encoder1 = torch.nn.Linear(5, emb_dim // 3)
        self.encoder2 = torch.nn.Linear(5, emb_dim // 3)
        self.encoder3 = torch.nn.Linear(1, emb_dim // 3)

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
        edge_encoded = self.encoder3(batch.x[:, :1])
        node_encoded1 = self.encoder1(batch.x[:, 1:6])
        node_encoded2 = self.encoder2(batch.x[:, 6:])
        batch.x = torch.cat([edge_encoded, node_encoded_features1, node_encoded_features2], dim=1)
        
        return batch

register_node_encoder('LinearNodeLG+LapPE', LinearNodeLGLapEncoder)
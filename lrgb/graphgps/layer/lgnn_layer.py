import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter, scatter_mean

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

   
class graph2linegraph(nn.Module):
    # Convert graph into a line graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self, variant=1):
        super().__init__()
        self.variant = variant
        
    def forward(self, batch):
        # Save original information in batch
        batch.shape = batch.x.shape
        batch.org_x = batch.x
        batch.org_edge_index = batch.edge_index
        batch.org_edge_attr = batch.edge_attr
        batch.org_batch = batch.batch
        if hasattr(batch, 'x0'):
            batch.org_x0 = batch.x0
        
        # Add bb edges
        if self.variant == 8:
            # print("FLAG - v8")
            node_count = torch.unique(batch.edge_index[0], return_counts=True)
            leaf_node = torch.where(node_count[1] == 1)[0]
            bb_edge = torch.stack((leaf_node, leaf_node), dim=0)
            batch.edge_index = torch.cat((batch.edge_index, bb_edge), dim=1)
            batch.edge_attr = torch.cat([batch.edge_attr, torch.zeros(leaf_node.shape[0], batch.edge_attr.shape[1], device=batch.y.device)], dim=0)
            indices = torch.argsort(batch.edge_index.T, dim=0)
            batch.edge_index = batch.edge_index.T[indices[:,0], :].T
            batch.edge_attr = batch.edge_attr[indices[:,0], :]
            
        
        lg_node_idx = batch.edge_index.T
        batch.lg_node_idx = lg_node_idx
        
        # NOTE: Line graph node feature 
        lg_x = batch.x[lg_node_idx]
        batch.x = torch.reshape(lg_x, (lg_x.shape[0], -1))
        lg_x = None
        if hasattr(batch, 'edge_attr'):
            batch.x = torch.stack([batch.x, batch.edge_attr.repeat(1,2)], dim=2).mean(dim=2)
            
        if hasattr(batch, 'x0'):
            lg_x0 = batch.x0[lg_node_idx]
            batch.x0 = torch.reshape(lg_x0, (lg_x0.shape[0], -1))
            lg_x0 = None


        # NOTE: line graph edge index
        # startNode, endNode = lg_node_idx[:, 0], lg_node_idx[:, 1]   
        lg_edge_idx = torch.nonzero((lg_node_idx[:, 1, None] == lg_node_idx[:, 0]) & (lg_node_idx[:, 0, None] != lg_node_idx[:, 1]))
        batch.edge_index = lg_edge_idx.T
        
        
        # NOTE: line graph edge attribute
        new_edge_idx = lg_node_idx[lg_edge_idx]
        lg_edge_attr = batch.org_x[new_edge_idx[:, 0, 1]].repeat(1,2)
        if hasattr(batch, 'edge_attr'):
            startEdge = new_edge_idx[:, 0]
            endEdge = new_edge_idx[:, 1]
            startIndices = torch.where(torch.all(batch.org_edge_index.T[:, None] == startEdge, dim=2))
            startEdgeAttr = batch.org_edge_attr[scatter(startIndices[0], startIndices[1])]
            endIndices = torch.where(torch.all(batch.org_edge_index.T[:, None] == endEdge, dim=2))
            endEdgeAttr = batch.org_edge_attr[scatter(endIndices[0], endIndices[1], dim=0)]
            if self.variant == 8 and startEdgeAttr.shape[0] < endEdgeAttr.shape[0]:
                startEdgeAttr = torch.cat([startEdgeAttr, torch.zeros(endEdgeAttr.shape[0] - startEdgeAttr.shape[0], startEdgeAttr.shape[1], device=batch.y.device)], dim=0)
            elif self.variant == 8 and startEdgeAttr.shape[0] > endEdgeAttr.shape[0]:
                endEdgeAttr = torch.cat([endEdgeAttr, torch.zeros(startEdgeAttr.shape[0] - endEdgeAttr.shape[0], endEdgeAttr.shape[1], device=batch.y.device)], dim=0)
            temp = torch.cat([startEdgeAttr, endEdgeAttr], dim=1)
            lg_edge_attr = torch.stack([lg_edge_attr, temp], dim=2).mean(dim=2)
            del startIndices
            del endIndices
            del startEdgeAttr
            del endEdgeAttr 
        batch.edge_attr = lg_edge_attr
        del lg_edge_idx
        
        if self.variant ==7:
            ptr = batch.ptr
            batch_size = batch.y.shape[0]
            new_batch = torch.where((batch.org_edge_index [0] >= ptr[:-1].unsqueeze(1)) & (batch.org_edge_index [0] < ptr[1:].unsqueeze(1)))
            batch.batch = new_batch[0]

        return batch

class linegraph2graph(nn.Module):
    # Convert line graph to graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self, variant=1):
        super().__init__()
        self.variant = variant
        
    def pad(self, tensor, originalShape):
        if originalShape[0] - tensor.shape[0] > 0:
            return torch.cat([tensor, torch.zeros(originalShape[0] - tensor.shape[0], originalShape[1], device=tensor.device)])
        else:
            return tensor
    
    def forward(self, batch):
        # Recover node feature
        shape = batch.shape
        frontNode = scatter_mean(batch.x, batch.lg_node_idx[:,0], dim=0)[:, shape[1]:]
        frontNode = self.pad(frontNode, shape)
        backNode = scatter_mean(batch.x, batch.lg_node_idx[:,1], dim=0)[:, :shape[1]]
        backNode = self.pad(backNode, shape)
        batch.x = torch.add(
            frontNode,
            backNode
        )
        
        # Recover edge feature
        shape = batch.org_edge_attr.shape
        frontEdge = scatter_mean(batch.edge_attr, batch.edge_index.T[:,0], dim=0)[:, shape[1]:]
        frontEdge = self.pad(frontEdge, shape)
        backEdge = scatter_mean(batch.edge_attr, batch.edge_index.T[:,1], dim=0)[:, :shape[1]]
        backEdge = self.pad(backEdge, shape)
        if self.variant == 8 and frontEdge.shape[0] > backEdge.shape[0]:
            backEdge = torch.cat([backEdge, torch.zeros(frontEdge.shape[0] - backEdge.shape[0], backEdge.shape[1], device=batch.y.device)], dim=0)
        elif self.variant == 8 and frontEdge.shape[0] < backEdge.shape[0]:
            frontEdge = torch.cat([frontEdge, torch.zeros(backEdge.shape[0] - frontEdge.shape[0], frontEdge.shape[1], device=batch.y.device)], dim=0)
        batch.edge_attr = torch.add(
            frontEdge,
            backEdge,
        )
        
        # Garbage Collecting 
        del frontNode
        del backNode
        del frontEdge
        del backEdge
        del shape
        
        if hasattr(batch, 'x0'):
            batch.x0 = batch.org_x0
        batch.edge_index = batch.org_edge_index
        
        return batch
    
    
class g2lg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, batch):
        batch.x = batch.lg_x
        batch.edge_index = batch.lg_edge_index
        batch.edge_attr = batch.lg_edge_attr
        
        return batch
    
    

class lg2graphNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.hdim = cfg.gnn.dim_inner // 3
        # self.linear = nn.Linear(self.dim * 2, self.dim)
    
    def zeroPadding(self, tensor1, tensor2):
        if tensor1.shape[0] > tensor2.shape[0]:
            zeros = torch.zeros(tensor1.shape[0] - tensor2.shape[0], tensor1.shape[1]).to(tensor1.device)
            tensor2 = torch.cat([tensor2, zeros])
        else:
            zeros = torch.zeros(tensor2.shape[0] - tensor1.shape[0], tensor1.shape[1]).to(tensor1.device)
            tensor1 = torch.cat([tensor1, zeros])
            
        return tensor1, tensor2
    
    def forward(self, batch):
        # Recover node feature
        mask = torch.cumsum(torch.cat([torch.zeros(1).to(batch.x.device), batch.org_graph_size]).type(torch.int64)[:-1], dim=0)
        padding = torch.repeat_interleave(mask, batch.ptr[1:] - batch.ptr[:-1])
        padded_lg_node_idx = batch.lg_node_idx + padding.repeat(2, 1).T
        
        linegraph2graphMapper = padded_lg_node_idx
        outgoingNode = scatter_mean(batch.x, linegraph2graphMapper[:,0], dim=0)
        incomingNode = scatter_mean(batch.x, linegraph2graphMapper[:,1], dim=0)
        
        if outgoingNode.shape[0] != incomingNode.shape[0]:
            outgoingNode, incomingNode = self.zeroPadding(outgoingNode, incomingNode)
        
        if cfg.gnn.lgvariant == 21:
            batch.x = torch.cat([incomingNode[:, :self.hdim*2], outgoingNode[:, self.hdim*2:]], dim=1)
        elif cfg.gnn.lgvariant == 22:
            batch.x = torch.cat([
                (incomingNode[:, :self.hdim] - outgoingNode[:, :self.hdim])/2,
                incomingNode[:, self.hdim:self.hdim*2],
                outgoingNode[:, self.hdim*2:]
            ], dim=1)
        else:
            batch.x = incomingNode - outgoingNode
        
        del mask
        del padding
        del padded_lg_node_idx
        del linegraph2graphMapper
        del outgoingNode
        del incomingNode
        
        return batch
    
class pcqmPostProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.hdim = cfg.gnn.dim_inner // 3
    
    def zeroPadding(self, tensor1, tensor2):
        if tensor1.shape[0] > tensor2.shape[0]:
            zeros = torch.zeros(tensor1.shape[0] - tensor2.shape[0], tensor1.shape[1]).to(tensor1.device)
            tensor2 = torch.cat([tensor2, zeros])
        else:
            zeros = torch.zeros(tensor2.shape[0] - tensor1.shape[0], tensor1.shape[1]).to(tensor1.device)
            tensor1 = torch.cat([tensor1, zeros])
            
        return tensor1, tensor2
    
    def forward(self, batch):
        # Recover node feature
        mask = torch.cumsum(torch.cat([torch.zeros(1).to(batch.x.device), batch.org_graph_size]).type(torch.int64)[:-1], dim=0)
        padding = torch.repeat_interleave(mask, batch.ptr[1:] - batch.ptr[:-1])
        padded_lg_node_idx = batch.lg_node_idx + padding.repeat(2, 1).T
        
        linegraph2graphMapper = padded_lg_node_idx
        outgoingNode = scatter_mean(batch.x, linegraph2graphMapper[:,0], dim=0)
        incomingNode = scatter_mean(batch.x, linegraph2graphMapper[:,1], dim=0)
        
        if outgoingNode.shape[0] != incomingNode.shape[0]:
            outgoingNode, incomingNode = self.zeroPadding(outgoingNode, incomingNode)
        
        if cfg.gnn.lgvariant == 21 or cfg.gnn.lgvariant == 30:
            batch.x = torch.cat([incomingNode[:, :self.hdim*2], outgoingNode[:, self.hdim*2:]], dim=1)
        elif cfg.gnn.lgvariant == 22:
            batch.x = torch.cat([
                (incomingNode[:, :self.hdim] + outgoingNode[:, :self.hdim])/2,
                incomingNode[:, self.hdim:self.hdim*2],
                outgoingNode[:, self.hdim*2:]
            ], dim=1)
        else:
            batch.x = incomingNode - outgoingNode
        
        del mask
        del padding
        del padded_lg_node_idx
        del linegraph2graphMapper
        del outgoingNode
        del incomingNode
        
        # NOTE: Adjust paddings for edge_index_labeled
        line_graph_size = torch.cumsum(torch.cat([
            torch.zeros(1).to(batch.x.device),
            (batch.ptr[1:] - batch.ptr[:-1])[:-1]
        ], dim=0), dim=0, dtype=torch.int64)
        edge_label_size = batch.edge_label_size
        lineGraphPadding = torch.repeat_interleave(line_graph_size, edge_label_size)
        
        graphSize = torch.cumsum(torch.cat([
            torch.zeros(1).to(batch.x.device),
            batch.org_graph_size[:-1],
        ]), dim=0, dtype=torch.int64)
        graphPadding = torch.repeat_interleave(graphSize, edge_label_size)
        
        # edge_index_labeled_new = batch.edge_index_labeled - lineGraphPadding.repeat(2, 1) + graphPadding.repeat(2, 1)
        edge_index_labeled_new = batch.edge_index_labeled - lineGraphPadding.repeat(2, 1)
        batch.edge_index_labeled = edge_index_labeled_new
        
        # NOTE: batch adjustment
        batch.ptr = torch.cumsum(torch.cat([
            torch.zeros(1).to(batch.x.device),
            batch.org_graph_size,
        ]), dim=0, dtype=torch.int64)
        
        batch.batch = torch.repeat_interleave(
            torch.arange(batch.org_graph_size.shape[0], device=batch.x.device),
            batch.org_graph_size
        )
        
        return batch
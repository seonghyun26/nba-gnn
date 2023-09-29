import os
import numpy as np
import torch
from torch.nn import functional as F
import dgl
from dgl import ops
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
import torch_geometric.transforms as T
from tqdm import tqdm
import pickle

class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False, nbtm=False, PE=False, PE_k=5):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        print('Preparing data...')
        path = 'DATA'
        if PE: 
            transform = T.AddLaplacianEigenvectorPE(k=PE_k, attr_name=None)
            if name in ['cora','citeseer','pubmed']:
                dataset = Planetoid(path, name, 'geom-gcn', transform=transform)
            elif name in ['texas','cornell', 'wisconsin']:
                dataset = WebKB(path, name, transform=transform)
            elif name in ['squirrel', 'chameleon']:
                dataset = WikipediaNetwork(path, name, transform=transform)
            elif name == 'actor':
                dataset =[]
                dataset_ = Actor(path, transform=transform)
                dataset.append(dataset_.data)
            print(f'PE with k = {PE_k}')
        else: 
            if name in ['cora','citeseer','pubmed']:
                dataset = Planetoid(path, name, 'geom-gcn')
            elif name in ['texas','cornell', 'wisconsin']:
                dataset = WebKB(path, name)
            elif name in ['squirrel', 'chameleon']:
                dataset = WikipediaNetwork(path, name)
            elif name == 'actor':
                dataset =[]
                dataset_ = Actor(path)
                dataset.append(dataset_.data)

        node_features = dataset[0].x
        labels = dataset[0].y
        edges= dataset[0].edge_index.T
        train_masks= dataset[0].train_mask.T.bool()
        val_masks= dataset[0].val_mask.T.bool()
        test_masks = dataset[0].test_mask.T.bool()

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)

        # if 'directed' not in name:
        #     graph = dgl.to_bidirected(graph)

        # if add_self_loops:
        graph = dgl.add_self_loop(graph)

        self.node_edges = graph.edges()[0].to(device) 

        if nbtm: 
            print(f'preparing for non-backtraking GNN.. ')
            graph, node_features, labels = self.nbtm_processing(name, PE, graph, node_features, labels)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        # node_features = self.augment_node_features(graph=graph,
                                                #    node_features=node_features,
                                                #    use_sgc_features=use_sgc_features,
                                                #    use_identity_features=use_identity_features,
                                                #    use_adjacency_features=use_adjacency_features,
                                                #    do_not_use_original_features=do_not_use_original_features)
        self.name = name
        self.device = device
        self.nbtm = nbtm 
        self.PE = PE 

        self.graph = graph.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    @staticmethod
    def augment_node_features(graph, node_features, use_sgc_features, use_identity_features, use_adjacency_features,
                              do_not_use_original_features):

        n = graph.num_nodes()
        original_node_features = node_features

        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = Dataset.compute_sgc_features(graph, original_node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_identity_features:
            node_features = torch.cat([node_features, torch.eye(n)], axis=1)

        if use_adjacency_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix().to_dense()
            node_features = torch.cat([node_features, adj_matrix], axis=1)

        return node_features

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features
    
    @staticmethod
    def nbtm_processing(name, PE, graph, node_features, labels):
        if not os.path.exists('nbtm'):
            os.makedirs('nbtm')
        file_name = os.path.join('nbtm', f'{name.replace("-", "_")}_{PE}.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                nbtm_graph = data['nbtm_graph']
                x = data['x']
                y = data['y']
        else: 
            node_edges = graph.edges()[0]
            neighbor_edges = graph.edges()[1]
            num_edges = graph.edges()[0].size(dim=0)

            node_x = torch.index_select(node_features, 0, node_edges)
            neighbor_x = torch.index_select(node_features, 0, neighbor_edges)
            x = torch.cat((node_x,neighbor_x), axis=1)
            y = labels

            new_edge = torch.empty((2,0), dtype=int)
            for j in tqdm(range(num_edges)):
                end_point = neighbor_edges[j]
                neighbors = torch.reshape((node_edges==end_point).nonzero(),(1,-1))
                j_edge = torch.cat((j*torch.ones_like(neighbors),neighbors), axis=0)
                new_edge = torch.cat((new_edge,j_edge), axis=1)

            nbtm_graph = dgl.graph((new_edge[0, :], new_edge[1, :]), num_nodes=len(x), idtype=torch.int)
            data = {}
            data['nbtm_graph'] =nbtm_graph 
            data['x'] =x
            data['y'] =y 
            with open(file_name, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return nbtm_graph, x, y 
            

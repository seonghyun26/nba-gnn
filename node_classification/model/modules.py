import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from torch.nn import init

class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res

        return x
    
class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x
    
class GatedGraphConvModule(nn.Module):
    def __init__(self, dim, n_steps, n_etypes=1, bias=True, **kwargs):
        super().__init__()
        self.dim = dim 
        self._in_feats = dim
        self._out_feats = dim
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(dim, dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain("relu")
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, etypes=None):
        with graph.local_scope():
            assert graph.is_homogeneous, (
                "not a homogeneous graph; convert it with to_homogeneous "
                "and pass in the edge type as argument"
            )
            if self._n_etypes != 1:
                assert (
                    etypes.min() >= 0 and etypes.max() < self._n_etypes
                ), "edge type indices out of range [0, {})".format(
                    self._n_etypes
                )

            zero_pad = feat.new_zeros(
                (feat.shape[0], self._out_feats - feat.shape[1])
            )
            feat = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                if self._n_etypes == 1 and etypes is None:
                    # Fast path when graph has only one edge type
                    graph.ndata["h"] = self.linears[0](feat)
                    graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")  # (N, D)
                else:
                    graph.ndata["h"] = feat
                    for i in range(self._n_etypes):
                        eids = (
                            torch.nonzero(etypes == i, as_tuple=False)
                            .view(-1)
                            .type(graph.idtype)
                        )
                        if len(eids) > 0:
                            graph.apply_edges(
                                lambda edges: {
                                    "W_e*h": self.linears[i](edges.src["h"])
                                },
                                eids,
                            )
                    graph.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")  # (N, D)
                feat = self.gru(a, feat)
            return feat

class GCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        x = ops.u_mul_e_sum(graph, x, norm_coefs)

        x = self.feed_forward_module(graph, x)

        return x


class SAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        message = ops.copy_u_mean(graph, x)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class GATModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.feed_forward_module(graph, x)

        return x


class GATSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x


class TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        message = ops.u_mul_e_sum(graph, values, attn_probs)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x

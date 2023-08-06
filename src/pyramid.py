from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian, add_self_loops
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv, ChebConv, ARMAConv

from proc.dataloader import parse_dataset

######################### Pyramid ###########################
def TRfilter(signal, kernel, sDv = None):
    return torch.matmul(kernel, signal)

class filtering(MessagePassing):
    r"""
    Args:
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, alpha: float, n_nodes:int, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        # self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, K,
                edge_weight: OptTensor = None, is_lowpass = True) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        Lap_index, Lap_weights = get_laplacian(edge_index, num_nodes=self.n_nodes, normalization='sym')

        h = x
        for k in range(K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            if is_lowpass:
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            else:
                x = self.propagate(Lap_index, x=x, edge_weight=Lap_weights, size=None)
            # as final output
            if is_lowpass:
                if k < K-1:
                    x = x * (1 - self.alpha) + self.alpha * h
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={K}, alpha={self.alpha})'



class PyGNN(nn.Module):
    def __init__(self, dataset, args):
        super(PyGNN, self).__init__()

        self.args = args
        # pyramid datasets
        self.S_subnode = [_[0] for _ in dataset.py_subg] # sub-nodes
        self.S_subedge = [_[1] for _ in dataset.py_subg] # subedges
        self.upsampl_ops = dataset.upsampl_ops        
        dataset, data = parse_dataset(args.dataname, dataset)
        
        # parameters
        self.n_bands = args.n_bands
        self.low_bands = args.low_bands
        self.aggregate = args.aggregate
        self.use_hp = args.use_hp
        self.dropout = nn.Dropout(args.dropout)
        self.input_drop = nn.Dropout(args.input_drop)
        self.K = [args.K for _ in range(self.n_bands)] 
        in_channels = dataset.n_feats 
        if self.aggregate=="concat":
            last_in_channels = args.hidden * args.n_bands 
        else:
            last_in_channels = args.hidden
        
        # gating
        if self.args.aggregate=="gate":
            self.lin_low_t = nn.Linear(args.hidden * self.low_bands, args.hidden)
            if args.n_bands - args.low_bands>0:
                self.lin_high_t = nn.Linear(args.hidden * (args.n_bands - args.low_bands), args.hidden)
        
        self.lin_low = nn.Linear(args.hidden, args.hidden)
        self.lin_high = nn.Linear(args.hidden, args.hidden)

        # self.layers = nn.ModuleList() 
        for b_idx in range(self.n_bands):
            if self.args.backbone=="SAGE":
                setattr(self, 'conv1_{}'.format(b_idx),
                        SAGEConv(in_channels, args.hidden, aggr = "max") ) 
            elif self.args.backbone=="ChebNet":
                setattr(self, 'conv1_{}'.format(b_idx),
                        ChebConv(in_channels, args.hidden, K= args.order) )
            elif self.args.backbone=="GAT":
                    setattr(self, 'conv1_{}'.format(b_idx),
                        GATConv(in_channels, args.hidden, heads=args.heads, dropout=0))
            elif self.args.backbone=="ARMA":
                    setattr(self, 'conv1_{}'.format(b_idx),
                        ARMAConv(in_channels, args.hidden, num_stacks=1, num_layers=1, shared_weights=True, dropout=0))
            elif self.args.backbone=="GCN":
                    setattr(self, 'conv1_{}'.format(b_idx),
                        GCNConv(in_channels, args.hidden) )
            else:
                assert 0, "backbone not implemented."
        self.filtering = filtering(alpha = args.alpha, n_nodes = dataset.n_nodes)  
        self.lin_out = nn.Linear(last_in_channels, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, feature, edge_index):
        device = feature.device
        feature = self.input_drop(feature)

        pyramid_layers = []
        for blk_idx in range(self.n_bands):
            ################# 
            conv_layer = getattr(self, 'conv1_{}'.format(blk_idx))
            x = conv_layer(feature, self.S_subedge[blk_idx])
            #################
            x = F.relu(x)
            x = self.dropout(x)
            # ######## filtering
            is_lowpass = (blk_idx<self.low_bands)
            filt_block = edge_index if is_lowpass else self.S_subedge[blk_idx] 
            x = self.filtering(x, filt_block, K = self.K[blk_idx], is_lowpass = is_lowpass)
            if self.args.use_upsampl and is_lowpass:   
                x = TRfilter(signal = x, kernel = self.upsampl_ops)
            pyramid_layers.append(x)

        if self.aggregate =="concat":
            x = torch.cat(pyramid_layers, dim=1) #{bz, n_subgraph * hunits}
        elif self.aggregate == "sum":
            x = torch.stack(pyramid_layers, dim=1) #{bz, n_subgraph, hunits}
            x = torch.sum(x, dim=1)
        elif self.aggregate == "gate":
            x_low = torch.cat(pyramid_layers[:self.low_bands], dim=1) # {bz, hunits*low_bands}
            x_high = torch.cat(pyramid_layers[self.low_bands:], dim=1) # {bz, hunits*high_bands}
            x_low_t = torch.tanh(self.lin_low_t(x_low)) # {bz, hunits}
            x_high_t = torch.relu(self.lin_high_t(x_high)) # {bz, hunits}  
            gate = torch.sigmoid(self.lin_low(x_low_t)+ self.lin_high(x_high_t)) # {bz, hunits}
            x = gate * x_low_t + (1-gate) * x_high_t

        x = self.lin_out(x)
        return F.log_softmax(x, dim=1)
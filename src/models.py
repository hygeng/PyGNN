import random
import math
import logging
import os.path as osp

import numpy as np
import scipy.sparse as sp
from scipy.special import comb
from scipy.sparse.linalg import lobpcg

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv, ChebConv, APPNP, ARMAConv

from module import *
import module
from proc.dataloader import DataLoader, random_planetoid_splits, parse_dataset

#########################   GPR-GNN  ###########################
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class GPRGNN_NC(nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN_NC, self).__init__()
        self.lin1 = nn.Linear(dataset.n_feats, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)
        
        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

#########################   BernNet  ###########################
class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP=F.relu(self.temp)
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #2I-L
        edge_index2, norm2= add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
        	x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
        	tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
        	x=tmp[self.K-i-1]
        	x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
        	for j in range(i):
        		x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

        	out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class BernNet_NC(nn.Module):
    def __init__(self,dataset, args):
        super(BernNet_NC, self).__init__()
        self.lin1 = nn.Linear(dataset.n_feats, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.m = nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(args.K)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

#########################   GCN   ###########################
class GCN_NC(nn.Module):
    def __init__(self, dataset, args):
        super(GCN_NC, self).__init__()
        self.conv1 = GCNConv(dataset.n_feats, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        #bn
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(args.hidden))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
#########################   multi-layer GCN   ###########################
class MULTI_GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MULTI_GCN, self).__init__()
        in_channels, hidden_channels, out_channels, num_layers, dropout = dataset.n_feats, \
            args.hidden, dataset.num_classes, args.n_layers, args.dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

#########################   Gheb   ###########################
class ChebNet_NC(nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet_NC, self).__init__()
        self.conv1 = ChebConv(dataset.n_feats, args.hidden, K=args.order)
        self.conv2 = ChebConv(args.hidden, dataset.num_classes,  K=args.order)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

#########################   GAT  ###########################
class GAT_NC(nn.Module):
    def __init__(self, dataset, args):
        super(GAT_NC, self).__init__()
        self.conv1 = GATConv(dataset.n_feats, args.hidden, \
            heads=args.heads, dropout=0)
        self.conv2 = GATConv(args.hidden * args.heads, dataset.num_classes, \
            heads=args.output_heads, concat=False, dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


#########################   GAT multi layer  ###########################
class MULTI_GAT(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MULTI_GAT, self).__init__()
        in_channels, hidden_channels, out_channels, num_layers,  heads, dropout = dataset.n_feats, \
            args.hidden, dataset.num_classes, args.n_layers, args.heads, args.dropout

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(nn.Linear(dataset.num_features, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                nn.Linear(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(nn.Linear(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()
    
    def forward(self, x, edge_index):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # x = x + self.skips[i](x)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def forward_mini(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
#########################   SAGE  ###########################
class SAGE_NC(torch.nn.Module):
    def __init__(self,  dataset, args):
        super(SAGE_NC, self).__init__()
        self.sage1 = SAGEConv(dataset.n_feats, args.hidden, aggr = "mean")  
        self.sage2 = SAGEConv(args.hidden, dataset.num_classes, aggr = "mean")
        self.dropout = args.dropout
        #bn
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(args.hidden))

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

#########################   Multi-layer graphsage  ###########################
class MULTI_SAGE(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MULTI_SAGE, self).__init__()
        in_channels, hidden_channels, out_channels, num_layers, dropout = dataset.n_feats, \
            args.hidden, dataset.num_classes, args.n_layers, args.dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=1)

#########################   APPNP  ###########################
class APPNP_NC(nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_NC, self).__init__()
        self.lin1 = nn.Linear(dataset.n_feats, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

#########################   MLP  ###########################
class MLP_NC(nn.Module):
    def __init__(self, dataset,args):
        super(MLP_NC, self).__init__()

        self.lin1 = nn.Linear(dataset.n_feats, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.dropout =args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    

#########################   ARMA  ###########################
class ARMA_NC(nn.Module):
    def __init__(self, dataset, args):
        super(ARMA_NC,self).__init__()
        self.conv1 = ARMAConv(dataset.n_feats, args.hidden, num_stacks=args.n_stacks,
                              num_layers=args.n_layers, shared_weights=True, dropout=args.dropout)

        self.conv2 = ARMAConv(args.hidden, dataset.num_classes, num_stacks=args.n_stacks,
                              num_layers=args.n_layers, shared_weights=True, dropout=args.dropout,
                              act=lambda x: x)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

#########################   UFG  ###########################
class UFG_NC(nn.Module):
    def __init__(self, dataset, args):
        super(UFG_NC, self).__init__()
        device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
        # prepare
        dataset, data = parse_dataset(args.dataname, dataset)
        num_nodes = data.x.shape[0]
        L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
        L = sp.coo_matrix((L[1].cpu(), (L[0][0, :].cpu(), L[0][1, :].cpu())), shape=(num_nodes, num_nodes))

        lobpcg_init = np.random.rand(num_nodes, 1)
        lambda_max, _ = lobpcg(L, lobpcg_init)
        lambda_max = lambda_max[0]

        # extract decomposition/reconstruction Masks
        FrameType = 'Haar' # args.FrameType
        if FrameType == 'Haar':
            D1 = lambda x: np.cos(x / 2)
            D2 = lambda x: np.sin(x / 2)
            DFilters = [D1, D2]
            RFilters = [D1, D2]
        elif FrameType == 'Linear':
            D1 = lambda x: np.square(np.cos(x / 2))
            D2 = lambda x: np.sin(x) / np.sqrt(2)
            D3 = lambda x: np.square(np.sin(x / 2))
            DFilters = [D1, D2, D3]
            RFilters = [D1, D2, D3]
        elif FrameType == 'Quadratic':  # not accurate so far
            D1 = lambda x: np.cos(x / 2) ** 3
            D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
            D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
            D4 = lambda x: np.sin(x / 2) ** 3
            DFilters = [D1, D2, D3, D4]
            RFilters = [D1, D2, D3, D4]
        else:
            raise Exception('Invalid FrameType')

        Lev = args.n_layers  # level of transform
        s = args.scale  # dilation scale
        n = args.order+1  # n - 1 = Degree of Chebyshev Polynomial Approximation
        J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
        r = len(DFilters)
        shrinkage = args.shrinkage
        sigma = args.sigma

        d = get_operator(L, DFilters, n, s, J, Lev)
        # enhance sparseness of the matrix operators (optional)
        # d[np.abs(d) < 0.001] = 0.0
        # store the matrix operators (torch sparse format) into a list: row-by-row
        self.d_list = list()
        for i in range(r):
            for l in range(Lev):
                self.d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

        self.GConv1 = UFGConv(dataset.n_feats, args.hidden, r, Lev, num_nodes, shrinkage=shrinkage, sigma=sigma)
        self.GConv2 = UFGConv(args.hidden, dataset.num_classes, r, Lev, num_nodes, shrinkage=shrinkage, sigma=sigma)
        
        self.dropout = nn.Dropout(args.dropout)
        self.reset_parameters()
    
    def forward(self, x, edge_index):
        d_list = self.d_list
        x = self.GConv1(x, d_list)
        x = self.dropout(x)
        x = self.GConv2(x, d_list)
        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


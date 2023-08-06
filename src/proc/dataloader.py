import math
import pickle
import os
import os.path as osp
import logging

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.utils import get_laplacian,  k_hop_subgraph
from torch_geometric.utils.undirected import is_undirected, to_undirected

from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

import sys
sys.path.append("./src/proc/") 
sys.path.append("./src/") 
from bases import cheby_coeff, ChebyshevApprox, cheby_op2
from utils import *

TOP_DIR = "./dataset/"
############################# data Loader ############################# 
def DataLoader(dataname, args=None):
    path = TOP_DIR
    if dataname in ['Cora', 'CiteSeer', 'Pubmed']:
        dataset = Planetoid(f"{path}/{dataname}/", dataname, transform=T.NormalizeFeatures(),  split = "full")
    elif dataname in ['chameleon', 'film', 'squirrel']:
        dataset = dataset_heterophily(root=path, name=dataname, transform=T.NormalizeFeatures())
        dataset = Data(**dataset.__dict__)
    else:
        raise ValueError(f'dataset {dataname} not supported in dataloader')
    return dataset

def parse_dataset(dataname, dataset):
    if dataname in ['chameleon', 'film', 'squirrel']:
        data = dataset["data"]
        dataset.n_feats = data.x.shape[1]
        dataset.num_classes = torch.unique(data.y).shape[0]
    else:
        dataset.n_feats = dataset.num_features
        data = dataset[0]
        
    dataset.n_nodes = data.x.shape[0]
    logging.info(f"{dataname} is {is_undirected(data.edge_index)} undirected graph")
    data.edge_index = to_undirected(data.edge_index)
    return dataset, data

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def PyGNNLoader(args, dataset, data, device):
    dataname = args.dataname
    path = f"{TOP_DIR}/{dataname}/{args.setname}.npy"
    S = np.load(path, allow_pickle = True)
    # upsampling operator
    order = args.upsampl_order
    upsampl_file = f"{TOP_DIR}/{dataname}/upsampl_{order}_{args.gamma:.2f}.pt"
    if osp.exists(upsampl_file):
        logging.info(f"load from: {upsampl_file}")
        dataset.upsampl_ops = torch.load(upsampl_file).to(device)
    else:
        upsampl_ops = cheby_upsampl_ops(args, data, data.edge_index, order = order)
        upsampl_ops = torch.from_numpy(upsampl_ops).float()
        logging.info(f"upsampling operator saved to: {upsampl_file}")
        torch.save(upsampl_ops, upsampl_file)
        dataset.upsampl_ops = upsampl_ops.to(device)
    def get_subg_edges(S, idx, device):
        S_subnode, S_subedge, S_submapping, S_submask = k_hop_subgraph(torch.LongTensor(S[idx]).to(device), \
                                                                num_hops = args.order, edge_index = data.edge_index)
        return to_device(S_subnode, device), to_device(S_subedge, device) #enclosing nodes, enclosing edges
    dataset.py_subg = [get_subg_edges(S, idx, device) for idx in range(len(S))]
    return dataset


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                p2raw=None,
                train_percent=0.01,
                transform=None, pre_transform=None):
        if name=='actor':
            name='film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name
        self._train_percent = train_percent
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        if not osp.isdir(root):
            os.makedirs(root)
        self.root = root
        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = float(self.data.train_percent)
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names
    @property
    def processed_file_names(self):
        return ['data.pt']
    def download(self):
        pass
    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
    def __repr__(self):
        return '{}()'.format(self.name)



############################# Upsampling Operator ############################# 
def cheby_upsampl_ops(args, data, e_index, order=5):
    # configs
    gamma = args.gamma
    # data
    n_nodes = data.num_nodes
    Lap_index, Lap_weights = get_laplacian(e_index, num_nodes=n_nodes, normalization='sym')
    L = sp.csr_matrix((Lap_weights, (Lap_index[0,:], Lap_index[1,:])), shape=(n_nodes, n_nodes))
    # ops
    g = lambda x: 1/(x * gamma + 1.0)
    c2 = cheby_coeff(g, order=order, quad_points=500, range1=0, range2=2) 
    # c = ChebyshevApprox(g, order = 5)
    upsampl_ops = cheby_op2(L, c2, range1=0, range2=2).toarray()
    return upsampl_ops

# ################## data utils #########################
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def mask_to_index(mask):
    index = torch.where(mask)[0]
    return index

def random_splits(data, seed=2022, dataname="None"):
    n_nodes = data.y.shape[0]
    index = np.arange(n_nodes)
    np.random.shuffle(index)
    
    n_train = round(n_nodes * 0.6)
    n_test = round(n_nodes * 0.2)
    
    train_idx= index[:n_train]
    val_idx = index[n_train: -n_test]
    test_idx= index[-n_test:]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    data.train_idx = torch.LongTensor(train_idx)
    data.val_idx =  torch.LongTensor(val_idx)
    data.test_idx =  torch.LongTensor(test_idx)
    # save data
    np.save(f"{TOP_DIR}/{dataname}/split/tmp_train_nids.npy",data.train_idx.numpy())
    np.save(f"{TOP_DIR}/{dataname}/split/tmp_valid_nids.npy", data.val_idx.numpy())
    np.save(f"{TOP_DIR}/{dataname}/split/tmp_test_nids.npy", data.test_idx.numpy())
    return data


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=2022, dataname="None"):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_batch = class_idx
        else:
            train_batch = rnd_state.choice(class_idx, percls_trn,replace=False)
        # print("train_batch: ",train_batch.shape)
        train_idx.extend(train_batch)
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    data.train_idx = torch.LongTensor(train_idx)
    data.val_idx =  torch.LongTensor(val_idx)
    data.test_idx =  torch.LongTensor(test_idx)
    # save data
    np.save(f"{TOP_DIR}/{dataname}/split/tmp_train_nids.npy",data.train_idx.numpy())
    np.save(f"{TOP_DIR}/{dataname}/split/tmp_valid_nids.npy", data.val_idx.numpy())
    np.save(f"{TOP_DIR}/{dataname}/split/tmp_test_nids.npy", data.test_idx.numpy())
    return data

def load_split_file(dataname, data, device):
    path = TOP_DIR
    train_nids = torch.from_numpy(np.load(f"{path}/{dataname}/split/train_nids.npy")).to(device)
    valid_nids = torch.from_numpy(np.load(f"{path}/{dataname}/split/valid_nids.npy")).to(device)
    test_nids  = torch.from_numpy(np.load(f"{path}/{dataname}/split/test_nids.npy") ).to(device)
    # logging.info(f"nids: {train_nids}, {valid_nids}, {test_nids}")
    data.train_mask = index_to_mask(train_nids,size=data.num_nodes)
    data.val_mask   = index_to_mask(valid_nids,size=data.num_nodes)
    data.test_mask  = index_to_mask(test_nids, size=data.num_nodes)
    # 
    data.train_idx = train_nids
    data.val_idx = valid_nids
    data.test_idx = test_nids
    return data


def citation_get_idx(data):
    data.train_idx = mask_to_index(data.train_mask)
    data.val_idx = mask_to_index(data.val_mask)
    data.test_idx = mask_to_index(data.test_mask)
    return data

def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)
    graph_id = 0
    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1
    return ret

# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

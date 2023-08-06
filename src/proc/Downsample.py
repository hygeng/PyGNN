from tqdm import tqdm
import time
import math
import argparse
import os.path as osp
import sys, getopt


import numpy as np
import scipy.sparse as sp

import torch
from torch_geometric.utils import get_laplacian

import sys
from bases import cheby_coeff, ChebyshevApprox, cheby_op2
from dataloader import DataLoader, random_planetoid_splits, parse_dataset, TOP_DIR


def Downsample(L, F, order = 5, is_eigen_decomp=False, argin=None):
    #  Input parameter (G is required, others are optional):
    #  L: Symmetric normalized graph Laplacian.
    #  F: # of vertices to be selected.
    #  bw: (Estimated) bandwidth the graph signals.
    #  nu: parameter controlling the width of the filter kernel.
    #  eigen_decomp: flag for eigendecomposition.
    #      1: Performs exact filtering (needs U and E)
    #      0: Approximates filter with Chebyshev polynomial approximation (needs order)
    #  argin: Inputs required for sensor selection.
    #      If eigen_decomp = 1, argin{1} = U and argin{2} = E
    #        U: Eigenvector matrix of L. i.e., GFT matrix.
    #        E: Eigenvalue matrix of L
    #      If eigen_decomp = 0, argin{1} = order
    #        order: Polynomial order.
    #  Output parameters:
    #  selected_nodes: Indices of vertex selected.
    #  T: Basis for reconstruction.
    #  
    #  References:
    #  A. Sakiyama, Y. Tanaka, T. Tanaka, and A. Ortega,
    #  "Eigendecomposition-free sampling set selection for graph signals," IEEE
    #  Transactions on Signal Processing, accepted.
    N = L.shape[0]

    A_tmp = (L - sp.diags(L.diagonal()) ).astype("bool")
    numedge = np.sum(A_tmp)/2
    p = numedge/N # edge probability
    n = F[-1]/N
    # # input
    # nu, bw = argin
    nu = 1.0
    bw = 2.0
    k = bw/N #bandwidth
    ## Preparing T
    if is_eigen_decomp:
        U, E = argin
        lmax=max(E)
        # g_E= math.exp(-nu*p*n*k*E/lmax)
        delta = 1.0
        g_E = 1.0/(delta + E)
        print(g_E)
        T_g_tmp1 = ( U.dot(np.diag(g_E)).dot(U.T) )
        print(T_g_tmp1.shape)
    else:
        range1, range2 = 0, 2
        delta = 1.0
        g = lambda x: 1/(x+delta)
        c = ChebyshevApprox(g, order)
        T_g_tmp1 = cheby_op2(L, c, range1, range2)
    print(T_g_tmp1.shape)
        
    def selection(selected, T_g_tmp):
        if len(selected)>0:
            T= T_g_tmp[:, selected].sum(1) # [N,1]
            T2=T.mean()-T 
            # T2[(T2<0)] = 0
            T2 = T2 * (T2>=0)
            T_g = T_g_tmp.dot(T2)  # N,N * N,1 = N,1
        else:
            T_g = T_g_tmp.sum(1)
        # 
        T_g[selected] = 0
        sensor  = np.argmax(T_g)
        return sensor
    def selection_incre(selected, T_g_tmp, sensor_prev, T):
        if len(selected)>0:
            # T= T_g_tmp[:, selected].sum(1) # [N.m]
            T = T + T_g_tmp[:, prev_sensor]
            T2=T.mean()-T.toarray()
            # T2[(T2<0)] = 0
            T2 = T2 * (T2>=0)
            T_g = T_g_tmp.dot(T2)
        else:
            T_g = T_g_tmp.sum(1)
        # 
        T_g[selected] = 0
        sensor  = np.argmax(T_g)
        return sensor, T
    ## SSS
    T_g_tmp=np.abs(T_g_tmp1)
    selected_nodes = []
    # 
    F = sorted(F)
    set_list = []
    tmp_list = []
    # T = np.zeros(T_g_tmp.shape[0])
    T = sp.csr_matrix((T_g_tmp.shape[0], 1))
    prev_sensor = []
    for i in tqdm(range(F[-1])):
        sensor, T = selection_incre(selected_nodes, T_g_tmp, prev_sensor, T)
        # sensor = selection(selected_nodes, T_g_tmp)
        prev_sensor = sensor
        # add
        selected_nodes.append(sensor)
        tmp_list.append(sensor)
        if (i+1) in F:
            set_list.append(np.array(tmp_list))
            tmp_list = []
    T = T_g_tmp1
    return np.asanyarray(set_list, dtype=object), T
    
def reconstruct(fn, S_set, T, order):
    T_k=T**order
    fn_recon= np.matmul(T_k[:,S_set], np.matmul( np.linalg.inv(T_k[S_set,:][:,S_set]), fn[S_set]))
    return fn_recon
    
    
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:d:e:", ["help=", "dataname=", "eigen="])
    except getopt.GetoptError:
        print('ERROR: python src/proc/Downsample.py -d <dataname>')
        sys.exit(2)
    print(opts)
    for opt, arg in opts:
        if opt == '-h':
            print('src/proc/Downsample.py -d <dataname> -e <is true>')
        if opt in ("-d", "--dataname"):
            dataname = arg
        if opt in ("-e", "--eigen"):
            # is_eigen_decomp = arg
            is_eigen_decomp = False

    is_eigen_decomp = False #(is_eigen_decomp=="1")
    dataset = DataLoader(dataname)
    dataset, data = parse_dataset(dataname, dataset)
    # stats
    n_nodes = data.x.shape[0]
    Lap = get_laplacian(data.edge_index, num_nodes=n_nodes, normalization='sym')
    L = sp.csr_matrix((Lap[1], (Lap[0][0, :], Lap[0][1, :]) ), shape=(n_nodes, n_nodes))
    # sampling
    F = (np.array([0.75, 0.875, 0.9375])*n_nodes).astype(int)
    argin = None
    S, T = Downsample(L, F, order =2, is_eigen_decomp=is_eigen_decomp, argin = argin)
    # save
    save_path = f"{TOP_DIR}/{dataname}/Downsample_set.npy"
    np.save(save_path, S)
    print(save_path, "\n")

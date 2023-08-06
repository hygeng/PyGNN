import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import copy
import sys, getopt, os
from dataloader import DataLoader, random_planetoid_splits, parse_dataset, TOP_DIR

#
n_bands = 4

def merge_sample(dataset, data):
    # stats
    n_nodes = data.x.shape[0]
    all_nodes = np.arange(n_nodes)

    prefix="Downsample"
    fname = f'{TOP_DIR}/{dataname}/{prefix}_set.npy'
    U = np.load(fname, allow_pickle=True)

    save_dir = f'{TOP_DIR}/{dataname}/'
    Lambset = []

    for i in range(len(U)):
        add_set = np.concatenate(U[:i+1]) 
        print(add_set)
        Lambset.append(add_set)
    for _ in range(n_bands - len(Lambset)):
        Lambset.append(all_nodes)

    for lset in Lambset:
        print(len(lset),end = "\t\t")
        
    np.save(save_dir + f"/pyramid.npy", np.asanyarray(Lambset,dtype=object))
    print(save_dir + f"/pyramid.npy")

def random_sample(dataset, data):
    prefix="Downsample"
    fname = f'{TOP_DIR}/{dataname}/{prefix}.npy'
    bands = np.load(fname, allow_pickle=True)

    # stats
    n_nodes = dataset[0].x.shape[0]
    all_nodes = np.arange(n_nodes)

    shuffle_list = all_nodes
    np.random.shuffle(shuffle_list)

    # n_subsets = [len(_) for _ in bands]
    n_subsets = (np.array([0.75, 0.875, 0.9375, 1.0])*n_nodes).astype(int)
    print("number of nodes in subsets: ",n_subsets)

    save_dir = f'{TOP_DIR}/{dataname}/'
    Lambset = []
    for i in range(n_bands):
        add_set = shuffle_list[:n_subsets[i]]
        Lambset.append(add_set)
    # Lambset.append(all_nodes)
    print("\n")
    for lset in Lambset:
        print(lset)
        
    for lset in Lambset:
        print(len(lset),end = "\t")
        
    np.save(f"{save_dir}/random.npy", np.asanyarray(Lambset,dtype=object))
    print(f"\nsave to: {save_dir}/random.npy")

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:d:s:", ["help=", "dataname=", "set="])
    except getopt.GetoptError:
        print('ERROR: python src/proc/Downsample.py -d <dataname>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('src/proc/Downsample.py -d <dataname>')
        if opt in ("-d", "--dataname"):
            dataname = arg
        if opt in ("-s", "--set"):
            set = arg

    # dataname = "chameleon"
    dataset = DataLoader(dataname)
    dataset, data = parse_dataset(dataname, dataset)
    
    if set=="random":
        random_sample(dataset, data)
    elif set=="Downsample":
        merge_sample(dataset, data)


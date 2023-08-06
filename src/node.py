import random
from tqdm import tqdm
import time
import sys
import pickle

# import seaborn as sns
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve

import torch
import torch.nn.functional as F


sys.path.append("./src/") 
from models import *
from utils import *
from pyramid import PyGNN
from proc.dataloader import TOP_DIR, DataLoader, random_planetoid_splits, parse_dataset, \
    citation_get_idx, PyGNNLoader, load_split_file, random_splits

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

epsilon = 1 - math.log(2)

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def compute_F1(preds, labels, n_classes):
    macro = {'F1': list(), 'Ac': list()}
    # preds = torch.argmax(torch.FloatTensor(predictions), dim=1).int()
    for cla in range(n_classes):
        tp = (labels[preds == cla] == cla).float().sum().item()
        fp = (labels[preds == cla] != cla).float().sum().item()

        tn = (labels[preds != cla] != cla).float().sum().item()
        fn = (labels[preds != cla] == cla).float().sum().item()

        # F1-score and Accuracy
        macro['F1'].append(np.divide(2 * tp , (2 * tp + fp + fn)))
        macro['Ac'].append(np.divide((tp + tn) , (tp + fp + tn + fn)))

    # average
    macro = {k: np.mean(v) for k, v in macro.items()}
    return macro

def compute_node_avg_auc(scores, labels, n_classes):
    auc_list,ap_list = [], []
    for nid in range(scores.shape[0]):
        labels_onehot = np.eye(n_classes)[labels[nid]]
        auc_list.append(roc_auc_score(labels_onehot.ravel(), scores[nid]))
        ap_list.append(average_precision_score(labels_onehot.ravel(), scores[nid]))
    auc = np.mean(auc_list)
    ap  = np.mean(ap_list)
    return auc, ap

def compute_auc(scores, labels, average, n_classes):
    '''
        labels: n_samples
        scores: n_samples, n_class (raw scores, or softmax probabilities)
    '''
    # n_class = int(scores.shape[1])
    labels_onehot = np.eye(n_classes)[labels]
    auc = average_precision_score(labels_onehot, scores, average=average)
    ap = roc_auc_score(labels_onehot, scores, average=average, multi_class='ovr')
    return auc, ap


def RunExp(args, dataset, data, Net, percls_trn, val_lb, RP):
    def train(model, optimizer, data, scheduler):
        model.train()
        optimizer.zero_grad()
        feat = data.x
        train_mask = data.train_idx
        edge_info = data.edge_index
        
        out = model(feat, edge_info)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        # optimize
        loss.backward()
        optimizer.step()
        if  args.use_scheduler: 
            scheduler.step()
        del out

    def test(model, data, eval_auc = True):
        model.eval()
        feat = data.x
        edge_info = data.edge_index
        
        logits, accs, F1s, losses, preds = model(feat, edge_info), [], [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
            if _=="test_mask" and eval_auc:
                macro = compute_F1(pred, data.y[mask], dataset.num_classes)
                F1 = macro['F1']
                # get auc, ap
                scores = logits[mask].detach().cpu()
                scores = F.softmax(scores, dim=1)
                scores = scores.numpy()
                # 
                n_class = scores.shape[1]
                labels = data.y[mask].detach().cpu().numpy()
                auc, ap = compute_auc(scores, labels, average="macro", n_classes=n_class)
            else:
                auc, ap, F1 = 0, 0, 0
        return accs, preds, losses, auc, ap, F1


    net = Net(dataset, args)
    model = net.to(device)
    if RP==0 and args.net=='Pyramid' and not args.test_mode:
        logging.info(f"{args.net} number of parameters: {count_parameters(model)}")

    #randomly split dataset
    if args.load_split: # given split data
        if RP==0 and not args.test_mode:
            logging.info(f"------ loading data split from {TOP_DIR}/{args.dataname}/split/")
        data = load_split_file(args.dataname, data, device)
    else:
        if args.dataname in ["Cora", "CiteSeer", "Pubmed"]:
            data = citation_get_idx(data)
        elif args.dataname in ["squirrel", "film", "chameleon"]:
            data = random_splits(data, args.seed, args.dataname)
        else:
            data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed, args.dataname)

    if args.net=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net =='BernNet':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.3, verbose=False)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    best_score = -float('inf')
    best_epoch = -1
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, scheduler)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        if epoch%5==0:
            [train_acc, val_acc, tmp_test_acc], preds, [
                train_loss, val_loss, tmp_test_loss], tmp_test_auc, tmp_test_ap, tmp_test_F1 = test(model, data, eval_auc=args.eval_auc)
            if not args.silent:
                logging.info(f"epoch: {epoch}, loss: {val_loss:.6f} train_acc: {train_acc*100:.4f}, val_acc: {val_acc*100:.4f}, tmp_test_acc: {tmp_test_acc*100:.4f}")
            if val_acc >= best_score:
                torch.save(model.state_dict(), f'{f_dir}/model.pth')
                best_epoch = epoch
                best_score = val_acc
                best_val_acc = val_acc
                # test
                test_acc = tmp_test_acc
                test_auc = tmp_test_auc
                test_ap  = tmp_test_ap
                test_F1 = tmp_test_F1
                if args.net =='BernNet':
                    TEST = net.prop1.temp.clone()
                    theta = TEST.detach().cpu()
                    theta = torch.relu(theta).numpy()
                else:
                    theta = args.alpha

            if epoch >= 0:
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                if args.early_stopping > 0 and epoch > args.early_stopping:
                    tmp = torch.tensor(
                        val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        logging.info(f'The sum of epochs:{epoch}')
                        break
    if args.test_mode:
        state_dict = torch.load(f'{args.load_dir}/model.pth', map_location = device)
        model.load_state_dict(state_dict, strict=False)
        # check parameters
        missing_keys = model.state_dict().keys() - state_dict.keys()
        unexpected_keys = state_dict.keys() - model.state_dict().keys()
        logging.info(f"Missing keys: {missing_keys}")
        logging.info(f"Unexpected keys: {unexpected_keys}")
        # model = model.to(device)
        [train_acc, val_acc, test_acc], preds, _, test_auc, test_ap, test_F1 = test(model, data, eval_auc=args.eval_auc)
    return test_acc, best_val_acc, time_run, test_auc, test_ap, test_F1, best_epoch


if __name__ == '__main__':
    args, f_dir = get_args()
    
    #10 fixed seeds for splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    logging.info(args)
    logging.info("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_NC
    elif gnn_name == 'MULTI_GCN':
        Net = MULTI_GCN
    elif gnn_name == 'GAT':
        Net = GAT_NC
    elif gnn_name == 'MULTI_GAT':
        Net = MULTI_GAT
    elif gnn_name == 'APPNP':
        Net = APPNP_NC
    elif gnn_name == 'ChebNet':
        Net = ChebNet_NC
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN_NC
    elif gnn_name == 'SAGE':
        Net = SAGE_NC
    elif gnn_name == 'MULTI_SAGE':
        Net = MULTI_SAGE
    elif gnn_name == 'BernNet':
        Net = BernNet_NC
    elif gnn_name =='MLP':
        Net = MLP_NC
    elif gnn_name =="ARMA":
        Net = ARMA_NC
    elif gnn_name =="UFG":
        Net = UFG_NC 
    elif gnn_name =="PyGNN":
        Net = PyGNN
    else:
        assert 0, "Not implemented"
    
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() and args.device>=0 else 'cpu')
    print("running on device: ", device)
    dataset = DataLoader(args.dataname, args)
    dataset, data = parse_dataset(args.dataname, dataset)
    if gnn_name =="PyGNN":
        dataset = PyGNNLoader(args, dataset, data, device)
    
    data = data.to(device)

    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))

    acc_results,  time_results, auc_results, ap_results, F1_results, val_acc_results = [], [], [], [], [], []

    for RP in range(args.runs):
        logging.info(f"runs: {RP}")
        args.seed=SEEDS[RP]
        # torch_seed(2022)
        # logging.info(f"seed: {args.seed}")
        test_acc, best_val_acc, time_run, test_auc, test_ap, test_F1, best_epoch = RunExp(args, dataset, data, Net, percls_trn, val_lb, RP)
        time_results.append(time_run)
        acc_results.append([test_acc, best_val_acc])
        # comprehensive evaluation metrics
        auc_results.append(test_auc)
        ap_results.append(test_ap)
        F1_results.append(test_F1)
        logging.info(f'End of run_{RP} \t test_acc: {test_acc:.4f} at epoch [{best_epoch}]')

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    test_acc_mean, val_acc_mean = np.mean(acc_results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(acc_results, axis=0)[0]) * 100
    val_acc_std = np.sqrt(np.var(acc_results, axis=0)[1]) * 100

    test_auc_mean = np.mean(auc_results) * 100
    test_auc_std = np.sqrt(np.var(auc_results)) * 100

    test_ap_mean = np.mean(ap_results) * 100
    test_ap_std = np.sqrt(np.var(ap_results)) * 100

    test_F1_mean = np.mean(F1_results) * 100
    test_F1_std = np.sqrt(np.var(F1_results)) * 100
    
    np.seterr(divide='ignore')
    acc_values=np.asarray(acc_results)[:,0]
    
    
    logging.info("-"*88)
    logging.info(f'[{gnn_name}] on dataset [{args.dataname}] for [Node Classification], in {args.runs} repeated experiment')
    logging.info(f'each run avg_time, each epoch avg_time: {np.divide(run_sum,args.runs):.3f}s\t{1000*np.divide(run_sum, epochsss):.3f}ms')
    logging.info(f'val acc     \ttest acc     \ttest auc     \ttest ap     \t test F1')
    logging.info(f'{val_acc_mean:.4f}±{val_acc_std:.4f}\t{test_acc_mean:.4f}±{test_acc_std:.4f}\t{test_auc_mean:.4f}±{test_auc_std:.4f}\t{test_ap_mean:.4f}±{test_ap_std:.4f}\t{test_F1_mean:.4f}±{test_F1_std:.4f}')

    logging.info(f"log dir: {f_dir}")
    
    # write result to file
    filename = os.path.join(f"./log/{args.dataname}/{args.net}/result.txt")
    print(f"save to: {filename}")
    with open(filename, 'a') as f: 
        f.write(f'{args.version}\t{val_acc_mean:.4f}±{val_acc_std:.4f}\t{test_acc_mean:.4f}±{test_acc_std:.4f}\t{test_auc_mean:.4f}±{test_auc_std:.4f}\t{test_ap_mean:.4f}±{test_ap_std:.4f}\t{test_F1_mean:.4f}±{test_F1_std:.4f}\t{f_dir}\n')


import datetime
import random, os
import argparse
import numpy as np
import torch

from datetime import datetime

import logging
import logging.config
from logging import handlers

def backup_code(save_path, version):
    os.system(F'rsync -ar -z --exclude-from=".gitignore" * {save_path}/{version}/')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='seeds for random splits.')
    
    parser.add_argument('--epochs', type=int, default=500, help='max epochs.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200,help='early.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--outdim', type=int, default=64, help='output units. for link prediction')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout for neural networks.')
    # data
    parser.add_argument('--input_drop', type=float, default=0, help='dropout for features')
    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--test_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--n_nodes', type=int, default=100, help='prob for ER graph')
    parser.add_argument('--prob', type=float, default=0.0, help='prob for ER graph')
    # run
    parser.add_argument('--dataname', type=str, default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=1, help='number of runs.')
    parser.add_argument('--net', type=str, default='Pyramid')
    parser.add_argument('--silent', action="store_true", help="silent in training")
    parser.add_argument('--load_split', action="store_true")
    parser.add_argument('--test_mode', action="store_true")
    parser.add_argument('--load_dir', type=str, default='') 
    parser.add_argument('--base_dir', type=str, default='', help="to which baseline log dir that compare the p-value") 
    parser.add_argument('--eval_auc', action="store_true",help="whether to evaluate AUC and AP")
    parser.add_argument('--use_scheduler', action="store_true")
    parser.add_argument('--version', type=str, default='default', help="model prefix")
    # GCN
    parser.add_argument('--self_loop', action="store_true", help="add self loop")
    # chebynet
    parser.add_argument('--order', type=int, default=2, help='polynomial order, for pyGNN and chebnet')
    # ARMA
    parser.add_argument("--n_layers",type=int, default=2,
	                help="number of layers. (ARMA, SAGE,...) Default is 2.")
    parser.add_argument("--n_stacks",type=int, default=2,
	                help="number of stacks. Default is 22.")    
    # APPNP
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.0, help='dropout for propagation layer.')
    # GPR-GNN
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],\
                default='PPR', help='initialization for GPRGNN.')
    # GAT
    parser.add_argument('--heads', default=4, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')
    # GWNN
    parser.add_argument("--filters",type=int, default=32,
	                help="Filters (neurons) in convolution. Default is 32.")
    parser.add_argument("--tolerance", type = float,default=10**-4,
	                help="Sparsification parameter. Default is 10^-4.")
    parser.add_argument("--scale", type=float, default=2.5,
	                help="Heat kernel scale length. Default is \
                        1.0;  dilation scale > 1 (default: 2.5) for UFG")
    # UFG
    parser.add_argument('--FrameType', type=str, default='Haar',
                    help='frame type (default: Haar)')
    parser.add_argument('--shrinkage', type=str, default='soft',
                        help='soft or hard thresholding (default: soft)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='standard deviation of the noise (default: 1.0)')
    # BernNet
    parser.add_argument('--Bern_lr', type=float, default=0.01, help=\
            'learning rate for BernNet propagation layer.')
    # PyGNN
    parser.add_argument('--use_upsampl', action="store_true")
    parser.add_argument('--upsampl_order', type=int, default=2, help='Number of bands')
    parser.add_argument('--use_hp', action="store_true")
    parser.add_argument('--setname', type=str, default='pyramid', help="method of subsampling")
    parser.add_argument('--aggregate', type=str, default='concat')
    parser.add_argument('--backbone', type=str, default='ChebNet', choices=["ChebNet", "SAGE"])
    parser.add_argument('--n_bands', type=int, default=4, help='Number of bands')
    parser.add_argument('--low_bands', type=int, default=4, help='number of low bands')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma upsampling operator')
    
    args = parser.parse_args()
    args.low_bands = min(args.low_bands, args.n_bands)

    # datetime object containing current date and time
    date =  datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'./log/{args.dataname}/{args.net}/{date}/'
    if args.test_mode:
        log_dir = args.load_dir
        args.eval_auc =True
        args.epochs=-1
        assert os.path.exists(f"{args.load_dir}/model.pth"), f"Ensure {args.load_dir}/model.pth exists "
        if args.base_dir != "":
            assert os.path.exists(f"{args.base_dir}/acc.npy"), f"Ensure {args.base_dir}/acc.npy exists "
        args.version = "test"
        os.makedirs(f'./log/{args.dataname}/{args.net}/saved', exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
        backup_code(log_dir, args.version)
    # 
    logging.config.fileConfig("./conf/logging.conf", defaults={"log_dir": log_dir, "version": args.version })
    return args, log_dir

def zip_files(f_dir):
    import scipy.misc as misc
    import shutil
    import zipfile

    top_folder= "./"
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(top_folder):
        for filename in filenames:
            if filename.split('\n')[0].split('.')[-1] in ['py', 'ipynb']:
                srczip.write(os.path.join(root, filename).replace(top_folder, ''))
    srczip.close()
    shutil.move('./src.zip',f_dir+'/src.zip')

def to_device(graph, device):
    if isinstance(graph, list):
        return [g.to(device) for g in graph]
    else:
        return graph.to(device)

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s-%(levelname)s: %(message)s'):
        # fmt='%(asctime)s-%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        self.logger = logging.getLogger()
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        # th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        fh = logging.FileHandler(filename, mode='a', encoding=None, delay=False)
        fh.setFormatter(format_str)
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        self.logger.addHandler(sh) 

def count_parameters(model):
    for name, parameters in model.named_parameters():
        logging.info(f"{name} : {parameters.size()}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def torch_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

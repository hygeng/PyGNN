# ######  cora
python src/node.py --dataname Cora  --lr 0.01 --dropout 0.5 --hidden 16 --net PyGNN --runs 10 --epochs 500 --device 0 --version final \
    --K 5 --alpha 0.3 --weight_decay 5e-4 --order 1 --setname pyramid --use_upsampl --use_scheduler --eval_auc --silent

# ###### citeseer
python src/node.py --dataname CiteSeer  --lr 0.01 --dropout 0.3 --hidden 16 --net PyGNN --runs 10 --epochs 500 --device 0 --version final \
        --K 0  --alpha 0.1 --weight_decay 5e-4 --order 2 --setname pyramid --use_upsampl --use_scheduler --eval_auc --silent 

# ###### pubmed
python src/node.py  --dataname Pubmed  --lr 0.01 --dropout 0.3 --hidden 16 --net PyGNN --runs 10 --epochs 500 --device 0 \
    --K 0  --alpha 0.2 --weight_decay 5e-4 --order 5 --setname pyramid --use_scheduler --use_upsampl --eval_auc --silent

# ###### chameleon
python src/node.py --dataname chameleon --net PyGNN  --lr 1e-3 --runs 10 --epochs 500 --n_bands 4 --low_bands 3 --version final \
     --dropout 0.5 --hidden 64 --K 5 --alpha 0.7 --weight_decay 5e-4 --order 4 --setname pyramid --use_scheduler \
    --aggregate sum --load_split --aggregate=gate  --device 0  --eval_auc --silent --use_upsampl

# ###### squirrel
python src/node.py --dataname squirrel --net PyGNN --lr 1e-3 --runs 10 --epochs 500 --n_bands 4 --low_bands 3  --version final  \
    --dropout 0.3 --hidden 64 --K 1 --alpha 0.7   --weight_decay 5e-4 --order 4 --setname pyramid --backbone SAGE\
    --use_scheduler --aggregate gate --eval_auc --load_split  --device 0 --silent

# ###### film 
python src/node.py --dataname film --net PyGNN --lr 1e-4 --runs 10 --epochs 500 --n_bands 4 --low_bands 3 --version final   \
     --dropout 0.5 --hidden 64  --K 0 --alpha 0.3  --weight_decay 5e-4 --order 2 --setname pyramid \
    --use_scheduler --aggregate gate --load_split --eval_auc --silent --device 0

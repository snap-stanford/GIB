#!/bin/bash 

## CORA

### GAT
python run_exp/run_nettack.py --exp_id=Cora-GAT --data_type=Cora-bool --model_type=GAT --beta1=-1 --beta2=-1 --struct_dropout_mode='\("standard",0.6\)' --seed=0 --gpuid=0

### GCN
python run_exp/run_nettack.py --exp_id=Cora-GCN --data_type=Cora-bool --model_type=GCN --beta1=-1 --beta2=-1 --seed=0 --gpuid=0

### GCNJaccard
python run_exp/run_nettack.py --exp_id=Cora-GCNJaccard --data_type=Cora-bool --model_type=GCNJaccard --beta1=-1 --beta2=-1 --latent_size=16 --lr=1e-2 --weight_decay=5e-4 --threshold=0.05 --seed=0 --gpuid=0

### RGCN
python run_exp/run_nettack.py --exp_id=Cora-RGCN --data_type=Cora-bool --model_type=RGCN --beta1=5e-4 --beta2=-1 --latent_size=64 --lr=1e-2 --weight_decay=5e-4 --gamma=0.3 --seed=0 --gpuid=0


# other datasets:
# --exp_id=Citeseer-GAT --data_type=citeseer-bool
# --exp_id=Pubmed-RGCN --data_type=Pubmed-bool

#!/bin/bash

# CORA

# GIB-CAT
python run_exp/run_nettack.py --exp_id=Cora-GIB-Cat --data_type=Cora-bool --model_type=GAT --beta1=0.001 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",1,3,2\)' --seed=0 --gpuid=0

# GIB-Bern
python run_exp/run_nettack.py --exp_id=Cora-GIB-Bern --data_type=Cora-bool --model_type=GAT --beta1=0.001 --beta2=0.01 --struct_dropout_mode='\("DNsampling","Bernoulli",0.1,0.5,"norm",2\)' --seed=0 --gpuid=0


#other datasets:
# --exp_id=Citeseer-GAT --data_type=citeseer-bool
# --exp_id=Pubmed-RGCN --data_type=Pubmed-bool
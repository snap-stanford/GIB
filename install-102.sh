#!/bin/bash
# author: stepp1s
# should not be used blindly. must read and understand possible errors
# instal pyg for cuda 10.2
# use cuda 10.2 because newer versions of cuda dont work with old pyg versions
# NOTE: the previous statement could benefit of more testing in docker envs

# check pytorch version, cuda and gpu availability
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# is nvcc 10.2?
nvcc --version

# install torch-geom reqs
python -m pip install torch-scatter==2.0.5 torch-sparse==0.6.10 torch-spline-conv==1.2.0 torch-cluster==1.5.7 -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
python -m pip install torch-geometric==1.5.0

python -c "import torch_sparse; import torch_geometric;"

# run GIB install process
git submodule init; git submodule update;
python -m pip install -r requirements.txt;
cd DeepRobust; pip install -e .;


echo "DONE!"
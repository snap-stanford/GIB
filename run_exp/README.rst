Run node feature attack experiments
==========================================================================

Here we list the commands to reproduce the results for node feature attack experiments. Each of the experiment in run_node.py loops over seed = 0, 1, 2, 3, 4. After running the experiments with the model saved under GIB/results/, use the script `experiments/GIB_node_analysis.ipynb <https://github.com/snap-stanford/GIB/blob/master/experiments/GIB_node_analysis.ipynb>`_ (Section 3) to perform node feature evasive attacks and obtain the results.

Cora
**********************

**Cora with GIB-Cat**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GAT --beta1=0.01 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",0.1,2,2\)' --gpuid=0

**Cora with AIB-Cat**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GAT --beta1=-1 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",0.1,2,2\)' --gpuid=0

**Cora with GIB-Bern**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GAT --beta1=0.001 --beta2=0.01 --struct_dropout_mode='\("DNsampling","Bernoulli",0.05,0.5,"norm",2\)' --gpuid=0

**Cora with AIB-Bern**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GAT --beta1=-1 --beta2=0.01 --struct_dropout_mode='\("DNsampling","Bernoulli",0.05,0.5,"norm",2\)' --gpuid=0

**Cora with GAT**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GAT --beta1=-1 --beta2=-1 --struct_dropout_mode='\("standard",0.6\)' --gpuid=0

**Cora with GCN**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GCN --beta1=-1 --beta2=-1 --gpuid=0

**Cora with GCNJaccard**:

.. code:: bash

    python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=GCNJaccard --beta1=-1 --beta2=-1 --latent_size=16 --lr=1e-2 --weight_decay=5e-4 --threshold=0.05 --gpuid=0

**Cora with RGCN**:

.. code:: bash

    python run_node.py --exp_id=node1.0 --data_type=Cora --model_type=RGCN --beta1=5e-4 --beta2=-1 --latent_size=64 --lr=1e-2 --weight_decay=5e-4 --gamma=0.3 --gpuid=0


Pubmed
**********************

**Pubmed with GIB-Cat**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GAT --beta1=0.001 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",1,3,2\)' --gpuid=0

**Pubmed with AIB-Cat**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GAT --beta1=-1 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",1,3,2\)' --gpuid=0

**Pubmed with GIB-Bern**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GAT --beta1=0.01 --beta2=0.01 --struct_dropout_mode='\("Nsampling","Bernoulli",0.05,0.5,"norm"\)' --gpuid=0


**Pubmed with AIB-Bern**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GAT --beta1=-1 --beta2=0.01 --struct_dropout_mode='\("Nsampling","Bernoulli",0.05,0.5,"norm"\)' --gpuid=0

**Pubmed with GAT**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GAT --beta1=-1 --beta2=-1 --struct_dropout_mode='\("standard",0.6\)' --gpuid=0

**Pubmed with GCN**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GCN --beta1=-1 --beta2=-1 --gpuid=0

**Pubmed with GCNJaccard**:

.. code:: bash
    
    python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=GCNJaccard --beta1=-1 --beta2=-1 --latent_size=16 --lr=1e-2 --weight_decay=5e-4 --threshold=0.05 --gpuid=0


**Pubmed with RGCN**:

.. code:: bash

    python run_node.py --exp_id=node1.0 --data_type=Pubmed --model_type=RGCN --beta1=5e-4 --beta2=-1 --latent_size=16 --lr=1e-2 --weight_decay=5e-4 --gamma=0.1 --gpuid=0


Citeseer
**********************

**Citeseer with GIB-Cat**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GAT --beta1=0.001 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",0.1,2,2\)' --gpuid=0


**Citeseer with AIB-Cat**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GAT --beta1=-1 --beta2=0.01 --struct_dropout_mode='\("DNsampling","multi-categorical-sum",0.1,2,2\)' --gpuid=0

**Citeseer with GIB-Bern**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GAT --beta1=0.1 --beta2=0.01 --struct_dropout_mode='\("DNsampling","Bernoulli",0.05,0.5,"norm",2\)' --gpuid=0

**Citeseer with AIB-Bern**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GAT --beta1=-1 --beta2=0.01 --struct_dropout_mode='\("DNsampling","Bernoulli",0.05,0.5,"norm",2\)' --gpuid=0

**Citeseer with GAT**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GAT --beta1=-1 --beta2=-1 --struct_dropout_mode='\("standard",0.6\)' --gpuid=0

**Citeseer with GCN**:

.. code:: bash

   python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GCN --beta1=-1 --beta2=-1 --gpuid=0

**Citeseer with GCNJaccard**:

.. code:: bash

    python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=GCNJaccard --beta1=-1 --beta2=-1 --latent_size=16 --lr=1e-2 --weight_decay=5e-4 --threshold=0.05 --gpuid=0

**Citeseer with RGCN**:

.. code:: bash

    python run_node.py --exp_id=node1.0 --data_type=citeseer --model_type=RGCN --beta1=5e-4 --beta2=-1 --latent_size=64 --lr=1e-2 --weight_decay=5e-4 --gamma=0.3 --gpuid=0


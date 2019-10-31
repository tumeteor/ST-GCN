# Introduction

The repo for temporal and spatial speed models, targets two scenarios:
- Speed imputation: increase the coverage of speed derived from GPS observations.
- Speed forecasting: predict future speed in the next 15 minute or 1 hour.

## The implemented models:
- Graph Convolutional Networks + LSTMs/CNNs
- (unfinished) Tensor Factorization

## Requirements
* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU

## Getting Started

* `preprocess.py` does feature extraction
* `trainer.py` train TGCN model
     - `python trainer.py --train`: train the data
     - `python trainer.py --test`: predict data and save in parquet format
     
## Deployment to Kubernetes

We can claim GPU(s) when creating a pod using the `k8s-trainer.yml` script. Currently, only one GPU
can be claimed for one pod.


## Feature extraction and data transformation

* The `preprocess.py` script expects data from [speed-model-dataset-spark](https://gitlab.mobilityservices.io/am/roam/realtime/speed-model-dataset-spark),
  with numpy tensors (each tensor corresponds to a grid), in the shape of `[num_nodes, 1, num_timesteps]`. Currently, all
  the downstream transformation is done using numpy / pandas on a single machine.
  
  The data is cached at:
   - aws-acc-001-1053-r1-master-data-science/speed/dataset/mytaxi/v3-timeseries-partitions-1hour/
  
  Then for every tensor, we enrich with features extracted from JURBEY and some basic time-senstive features, 
  window-slicing it to new tensor of shape `[num_nodes, num_features, num_look_back_step, num_timesteps]` for `data`,
  and `[num_nodes, num_features, num_look_ahead_step, num_timesteps]` for `target`, similarly for `mask`.
  The window-sliced data is cached at:
   -  s3://aws-acc-001-1053-r1-master-data-science/speed/dataset/mytaxi/features_400/
   
   NOTE: the grid-based clustering is done separately and the mapping is in the `cluster-mapping.csv` file.
   The adjacency matrices and edge list wrt. cluster (grid) IDs are cached in:
   -  s3://aws-acc-001-1053-r1-master-data-science/speed/dataset/mytaxi/adjs/
  

 


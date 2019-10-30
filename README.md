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
 


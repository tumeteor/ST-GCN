from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
from pytorch_lightning import Trainer
import argparse
import logging
import yaml
import torch
import scipy
import os
import pickle as pkl
import pandas as pd
from torch.utils.data import DataLoader

from src.configs.configs import TGCN as TGCNConfig
from src.configs.configs import Data as DataConfig
from src.data_loader.reader import read_cluster_mapping
from src.data_loader.tensor_dataset import GraphTensorDataset
from src.logs import get_logger_settings, setup_logging
from src.models.tgcn.temporal_spatial_model import TGCN
from src.utils.sparse import dense_to_sparse, sparse_scipy2torch

with open("configs/configs.yaml") as ymlfile:
    cfg = yaml.safe_load(ymlfile)['DataConfig']


def get_datasets():
    mapping = read_cluster_mapping()
    cluster_idx_ids = dict()
    datasets = list()
    adjs = list()
    edgelists = list()
    cluster_idx = 0
    for cluster_id in mapping:
        # cache them in h5
        if not os.path.exists(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5")):
            # some clusters do not exist in the cache folder, ignore them.
            continue

        adj = scipy.sparse.load_npz(os.path.join(cfg['save_dir_adj'], f"cluster_id={cluster_id}.npz"))
        adjs.append(adj)
        edgelist = pkl.load(open(os.path.join(cfg['save_dir_adj'], f"cluster_id={cluster_id}.edgelist"), 'rb'))
        edgelists.append(edgelist)
        datasets.append(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}/"))
        cluster_idx_ids[cluster_idx] = cluster_id
        cluster_idx += 1
    return datasets, adjs, cluster_idx_ids, edgelists


def train():
    datasets, adjs, cluster_idx_ids, _ = get_datasets()
    # PyTorch summarywriter with a few bells and whistles
    exp = Experiment(save_dir=cfg['save_dir_model'])
    checkpoint_callback = ModelCheckpoint(
        filepath=cfg['save_dir_checkpoints'],
        save_best_only=True,
        verbose=True,
        monitor='avg_val_mae',
        mode='min'
    )

    # pass in experiment for automatic tensorboard logging.
    trainer = Trainer(experiment=exp,
                      max_nb_epochs=TGCNConfig.max_nb_epochs,
                      train_percent_check=TGCNConfig.train_percent_check,
                      checkpoint_callback=checkpoint_callback,
                      gpus=1) if torch.cuda.is_available() else \
        Trainer(experiment=exp,
                max_nb_epochs=TGCNConfig.max_nb_epochs,
                train_percent_check=TGCNConfig.train_percent_check,
                checkpoint_callback=checkpoint_callback)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TGCN(input_dim=TGCNConfig.input_dim,
                 hidden_dim=TGCNConfig.hidden_dim,
                 layer_dim=TGCNConfig.layer_dim,
                 output_dim=TGCNConfig.output_dim,
                 adjs=adjs,
                 datasets=datasets,
                 cluster_idx_ids=cluster_idx_ids,
                 device=device)
    model = model.to(device)
    trainer.fit(model)


def load_model(weights_path, adjs, datasets, cluster_idx_ids, device=None):
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    model = TGCN(input_dim=TGCNConfig.input_dim,
                 hidden_dim=TGCNConfig.hidden_dim,
                 layer_dim=TGCNConfig.layer_dim,
                 output_dim=TGCNConfig.output_dim,
                 adjs=adjs,
                 datasets=datasets,
                 cluster_idx_ids=cluster_idx_ids,
                 device=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.on_load_checkpoint(checkpoint)
    model.freeze()
    return model


def get_data_loader(datasets, adjs, cluster_idx_ids, mode):
    if mode == "train":
        time_steps = DataConfig.train_num_steps
    elif mode == "valid":
        time_steps = DataConfig.valid_num_steps
    else:
        time_steps = DataConfig.test_num_steps
    ds = GraphTensorDataset(datasets, adj_list=adjs,
                            mode=mode,
                            cluster_idx_ids=cluster_idx_ids,
                            time_steps=time_steps)
    return DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)


def test():
    datasets, adjs, cluster_idx_ids, edgelists = get_datasets()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(cfg['save_dir_checkpoints'],
                       adjs=adjs, cluster_idx_ids=cluster_idx_ids, datasets=datasets, device=device)
    model = model.to(device)
    adjs = [sparse_scipy2torch(adj) for adj in adjs]
    train_dataloader = get_data_loader(datasets, adjs, cluster_idx_ids, mode="train")
    valid_dataloader = get_data_loader(datasets, adjs, cluster_idx_ids, mode="valid")
    test_dataloader = get_data_loader(datasets, adjs, cluster_idx_ids, mode="test")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dl_idx = 0
    for dl in (train_dataloader, valid_dataloader, test_dataloader):
        if dl_idx == 0:
            time_steps = DataConfig.train_num_steps
            base_steps = 0
            mode = "train"
        elif dl_idx == 1:
            time_steps = DataConfig.valid_num_steps
            base_steps = DataConfig.train_num_steps
            mode = "valid"
        else:
            time_steps = DataConfig.test_num_steps
            base_steps = DataConfig.train_num_steps + DataConfig.test_num_steps
            mode = "test"

        speed_tile = {}
        prev_gidx = 0
        with torch.no_grad():
            for batch_nb, batch in enumerate(dl):
                time_step = batch_nb % time_steps + base_steps
                graph_idx = int(batch_nb / time_steps)
                if graph_idx > prev_gidx:
                    df = pd.DataFrame.from_dict(speed_tile)
                    df.columns = list(map(str, range(time_steps)))
                    df['segment_id'] = edgelists[prev_gidx]
                    for n, col in enumerate(['from_node', 'to_node']):
                        df[col] = df['segment_id'].apply(lambda l: l[n])
                    df = df.drop('segment_id', axis=1)
                    df.to_parquet(f"{graph_idx}_{mode}_generated_speed.parquet", compression="snappy")
                    speed_tile = {}
                if batch_nb is None:
                    continue

                batch = [b.to(device) for b in batch]
                x, y, adj, mask = batch
                mask = mask.float()
                adj = dense_to_sparse(adj.squeeze(dim=0)).float()
                x = x.squeeze(dim=0)
                x = x.permute(0, 2, 1)

                y_hat = model.forward(x, adj).squeeze(dim=0)
                # fuse y and y_hat: take ground-truth speeds
                fused_y = mask * y.float() + (1 - mask) * y_hat.float()
                speed_tile[time_step] = fused_y
                prev_gidx = graph_idx
        dl_idx += 1


if __name__ == "__main__":
    log_setting = get_logger_settings(logging.INFO)
    setup_logging(log_setting)

    parser = argparse.ArgumentParser(description='Speed model training / serving.')
    parser.add_argument('--train', dest='train_mode', action='store_true')
    parser.add_argument('--test', dest='inference_mode', action='store_false')
    parser.set_defaults(train_mode=True)
    args = parser.parse_args()

    if args.train_mode:
        train()
    else:
        test()








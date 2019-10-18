import torch

import scipy
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
from pytorch_lightning import Trainer
import argparse
import json
import logging
import yaml
from src.configs.db_config import Config
from src.configs.configs import TGCN as TGCNConfig
from src.data_loader.reader import read_jurbey, read_cluster_mapping
from src.logs import get_logger_settings, setup_logging
from src.models.tgcn.temporal_spatial_model import TGCN

with open("src/configs/configs.yaml") as ymlfile:
    cfg = yaml.load(ymlfile)['DataConfig']

if __name__ == "__main__":
    db_cfg = Config()
    parser = argparse.ArgumentParser(description='Compute Weight for Routing Graph')
    parser.add_argument('--artifact', type=str, help='path to the start2jurbey artifact')
    args = parser.parse_args()
    log_setting = get_logger_settings(logging.INFO)
    setup_logging(log_setting)

    if args.artifact:
        artifact_path = args.artifact
    else:
        artifact_path = db_cfg.INPUT_PATH

    with open(artifact_path, 'r') as f:
        message = json.load(f)

    logging.info('\u2B07 Getting Jurbey File...')
    # g = read_jurbey_from_minio(message['bucket'], message['jurbey_path'])
    g = read_jurbey()
    logging.info("\u2705 Done loading Jurbey graph.")

    mapping = read_cluster_mapping()
    cluster_idx_ids = dict()
    datasets = list()
    adjs = list()
    cluster_idx = 0
    for cluster_id in mapping:
        # cache them in h5
        if not os.path.exists(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5")):
            # some clusters do not exist in the cache folder, ignore them.
            continue

        adj = scipy.sparse.load_npz(os.path.join(cfg['save_dir_adj'], f"cluster_id={cluster_id}.npz"))

        adjs.append(adj)

        datasets.append(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}/"))
        cluster_idx_ids[cluster_idx] = cluster_id
        cluster_idx += 1

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
                      checkpoint_callback=checkpoint_callback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

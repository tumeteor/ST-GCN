import os

from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
from pytorch_lightning import Trainer
import argparse
import json
import logging

from src.config import Config
from src.data_loader.reader import read_jurbey, read_cluster_mapping, get_adj_from_subgraph
from src.logs import get_logger_settings, setup_logging
from src.data_loader.datasets import DatasetBuilder
from src.models.tgcn.temporal_spatial_model import TGCN

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description='Compute Weight for Routing Graph')
    parser.add_argument('--artifact', type=str, help='path to the start2jurbey artifact')
    args = parser.parse_args()
    log_setting = get_logger_settings(logging.INFO)
    setup_logging(log_setting)

    if args.artifact:
        artifact_path = args.artifact
    else:
        artifact_path = cfg.INPUT_PATH

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
        if not os.path.exists(f"data/test_cache/cluster_id={cluster_id}.hdf5"):
                # some clusters do not exist in the cache folder, ignore them.
                continue
        db = DatasetBuilder(g=g)
        edges, df = db.load_speed_data(file_path=f"data/clusters/cluster_id={cluster_id}/")
        if len(edges) < 100:
            # remove too small clusters
            continue

        adj, _ = get_adj_from_subgraph(cluster_id=cluster_id, g=g, edges=edges)
        adjs.append(adj)

        datasets.append("data/test_cache/cluster_id={cluster_id}.hdf5")
        cluster_idx_ids[cluster_idx] = cluster_id
        cluster_idx += 1

    # PyTorch summarywriter with a few bells and whistles
    exp = Experiment(save_dir='data/models/tgcn/')
    checkpoint_callback = ModelCheckpoint(
        filepath='data/models/tgcn/checkpoints/',
        save_best_only=True,
        verbose=True,
        monitor='avg_val_mae',
        mode='min'
    )

    # pass in experiment for automatic tensorboard logging.
    trainer = Trainer(experiment=exp, max_nb_epochs=45, train_percent_check=1,
                      checkpoint_callback=checkpoint_callback)

    model = TGCN(input_dim=29, hidden_dim=29, layer_dim=2, output_dim=1, adjs=adjs,
                 datasets=datasets, cluster_idx_ids=cluster_idx_ids)

    trainer.fit(model)

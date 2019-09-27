import os

import h5py
import numpy as np
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

    data = list()
    targets = list()
    adjs = list()
    masks = list()
    for cluster_id in mapping:
        if cluster_id != 95 and cluster_id != 96: continue
        db = DatasetBuilder(g=g)
        edges, df = db.load_speed_data(file_path=f"data/clusters/cluster_id={cluster_id}/")
        if len(edges) < 100: continue

        adj, L = get_adj_from_subgraph(cluster_id=cluster_id, g=g, edges=edges)
        # cache them in h5
        if os.path.exists(f"data/cache/cluster_id={cluster_id}.hdf5"):
            h = h5py.File(f"data/cache/cluster_id={cluster_id}.hdf5", "r")
        else:
            _data, target, mask = db.construct_batches(df, L=L)
            h = h5py.File(f"data/cache/cluster_id={cluster_id}.hdf5", "w")
            data_group = h.create_group(name="data")
            for k, v in _data.items():
                data_group.create_dataset(k, data=v, chunks=True, compression='gzip')
            target_group = h.create_group(name="target")
            for k, v in target.items():
                target_group.create_dataset(k, data=v, chunks=True, compression='gzip')
            mask_group = h.create_group(name="mask")
            for k, v in mask.items():
                mask_group.create_dataset(k, data=v, chunks=True, compression='gzip')

        data.append(h["data"])
        targets.append(h["target"])
        masks.append(h["mask"])
        adjs.append(adj)
    datasets = {"train": list(), "valid": list(), "test": list()}
    mask_dict = {"train": list(), "valid": list(), "test": list()}

    for gidx in range(len(adjs)):
        datasets['train'] += [(x, t) for x, t in zip(
            np.split(np.array(data[gidx]['train']), len(np.array(data[gidx]['train'])), axis=0),
            np.split(np.array(targets[gidx]['train']), len(np.array(targets[gidx]['train'])), axis=0))]

        datasets['valid'] += [(x, t) for x, t in zip(
            np.split(np.array(data[gidx]['valid']), len(np.array(data[gidx]['valid'])), axis=0),
            np.split(np.array(targets[gidx]['valid']), len(np.array(targets[gidx]['valid'])), axis=0))]

        datasets['test'] += [(x, t) for x, t in zip(
            np.split(np.array(data[gidx]['test']), len(np.array(data[0]['test'])), axis=0),
            np.split(np.array(targets[gidx]['test']), len(np.array(targets[gidx]['test'])), axis=0))]

        mask_dict['train'] += [x for x in np.split(np.array(masks[gidx]['train']),
                                                                    len(np.array(masks[gidx]['train'])), axis=0)]
        mask_dict['valid'] += [x for x in np.split(np.array(masks[gidx]['valid']),
                                                                    len(np.array(masks[gidx]['valid'])), axis=0)]
        mask_dict['test'] += [x for x in np.split(np.array(masks[gidx]['test']),
                                                                   len(np.array(masks[gidx]['test'])), axis=0)]

    # PyTorch summarywriter with a few bells and whistles
    exp = Experiment(save_dir="../data/models/tgcn/")

    # pass in experiment for automatic tensorboard logging.
    trainer = Trainer(experiment=exp, max_nb_epochs=45, train_percent_check=1)

    model = TGCN(input_dim=29, hidden_dim=29, layer_dim=2, output_dim=1, adjs=adjs,
                 datasets=datasets, masks=mask_dict)

    trainer.fit(model)

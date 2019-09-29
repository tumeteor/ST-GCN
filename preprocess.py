import os

import h5py
import argparse
import json
import logging

from src.config import Config
from src.data_loader.reader import read_jurbey, read_cluster_mapping, get_adj_from_subgraph
from src.logs import get_logger_settings, setup_logging
from src.data_loader.datasets import DatasetBuilder

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
        db = DatasetBuilder(g=g)
        edges, df = db.load_speed_data(file_path=f"data/clusters/cluster_id={cluster_id}/")
        if len(edges) < 100: continue

        adj, L = get_adj_from_subgraph(cluster_id=cluster_id, g=g, edges=edges)
        # cache them in h5
        if not os.path.exists(f"data/cache/cluster_id={cluster_id}.hdf5"):
            _data, target, mask = db.construct_batches(df, L=L, memmap=True)
            with h5py.File(f"data/cache/cluster_id={cluster_id}.hdf5", "w") as h:
                data_group = h.create_group(name="data")
                for k, v in _data.items():
                    data_group.create_dataset(k, data=v, chunks=True, compression='gzip')
                target_group = h.create_group(name="target")
                for k, v in target.items():
                    target_group.create_dataset(k, data=v, chunks=True, compression='gzip')
                mask_group = h.create_group(name="mask")
                for k, v in mask.items():
                    mask_group.create_dataset(k, data=v, chunks=True, compression='gzip')

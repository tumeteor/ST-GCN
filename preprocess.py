import os
import scipy
import h5py
import pickle as pkl
import argparse
import json
import logging
from src.configs.db_config import Config
from src.data_loader.reader import read_jurbey, read_cluster_mapping, get_adj_from_subgraph
from src.logs import get_logger_settings, setup_logging
from src.data_loader.datasets import DatasetBuilder
from src.module import DATACONFIG_GETTER

cfg = DATACONFIG_GETTER()

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

    data = []
    targets = []
    adjs = []
    masks = []
    for cluster_id in mapping:
        db = DatasetBuilder(g=g)
        edges, df = db.load_speed_data(file_path=os.path.join(cfg['all_cluster_path'], f"cluster_id={cluster_id}"))
        if len(edges) < 100:
            continue
        if not os.path.exists(os.path.join(cfg['save_dir_adj'], f"cluster_id={cluster_id}.edgelist")):
            with open(os.path.join(cfg['save_dir_adj'], f"cluster_id={cluster_id}.edgelist"), 'wb') as f:
                pkl.dump(edges, f)

        adj, L = get_adj_from_subgraph(cluster_id=cluster_id, g=g, edges=edges)

        if not os.path.exists(os.path.join(cfg['all_cluster_path'], f"cluster_id={cluster_id}.npz")):
            scipy.sparse.save_npz(os.path.join(cfg['all_cluster_path'], f"cluster_id={cluster_id}.npz"), adj)

        # cache them in h5
        if not os.path.exists(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5")):
            _data, target, mask = db.construct_batches(df, L=L, memmap=True)
            with h5py.File(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5"), "w") as h:
                data_group = h.create_group(name="data")
                for k, v in _data.items():
                    data_group.create_dataset(k, data=v, chunks=True, compression='gzip')
                target_group = h.create_group(name="target")
                for k, v in target.items():
                    target_group.create_dataset(k, data=v, chunks=True, compression='gzip')
                mask_group = h.create_group(name="mask")
                for k, v in mask.items():
                    mask_group.create_dataset(k, data=v, chunks=True, compression='gzip')


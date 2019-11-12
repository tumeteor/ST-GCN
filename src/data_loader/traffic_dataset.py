from src.data_loader.reader import read_cluster_mapping, get_adj_from_subgraph
from src.utils.tensor import is_finite
from src.data_loader.features import (
            max_speed_cats,
            num_lanes_cats,
            road_class,
            surface
        )
from fastparquet import ParquetFile
from itertools import product
import torch
import h5py
import yaml
import glob
import os

with open("configs/configs.yaml") as ymlfile:
    cfg = yaml.load(ymlfile)['DataConfig']

def get_encoder(*args):
    return {c: i for i, c in enumerate(product(*args))}

def one_hot(x, size):
    vec = torch.zeros(size, dtype=torch.float)
    vec[x] = 1
    return vec

class TrafficDataset(torch.utils.data.IterableDataset):

    def __init__(self, G, nb_buckets = 24, slice_size = 30):
        super().__init__()
        self.G = G
        self.nb_buckets = nb_buckets
        self.slice_size = slice_size
        self.features_encoder = get_encoder(road_class, range(nb_buckets))
        self.road_enc = get_encoder(max_speed_cats, num_lanes_cats)
        self.mapping = iter(read_cluster_mapping())

    def __next__(self):
        # TODO Select correct time window to load
        # TODO if new_cluster_test: self._load_new_cluster
        # TODO Slice data to the needed window
        if (self.i + self.slice_size) % (self.X.size(0) - 1) == 0:
            self._load_new_cluster()
        indices = [(i, i + (self.slice_size)) for i
                   in range(X.shape[2] - (
                    num_lookback + num_lookahead) + 1)]
        start = self.i % self.X.size(0)
        end = start + self.slice_size

        features = self.X[start: start + self.slice_size-1, :, :]
        target = self.X[start + self.slice_size, :, 0]
        mask = self.mask[:, 0, start + self.slice_size-1:end]

        self.i += 1
        return features, target, mask

    def __iter__(self):
        # TODO Check worker's state
        # TODO Select clusters to work on
        # Initialize state
        self._load_new_cluster()
        self.i = 0
        return self

    def _load_cluster_data(self, df, L):
        id_to_idx = {x: i for i, x in enumerate(L.nodes())}

        df_filled = df.loc[:, df.columns != 'from_node']
        df_filled = df_filled.loc[:, df_filled.columns != 'to_node']

        df_filled = df_filled.fillna(df_filled.mean())
        df_filled = df_filled.fillna(13.8)
        total_t_steps = df.shape[1] - 2
        speed_columns = [str(i) for i in range(total_t_steps)]
        df.columns = ['from_node', 'to_node'] + speed_columns
        df_filled.columns = speed_columns
        df_filled['from_node'] = df['from_node']
        df_filled['to_node'] = df['to_node']
        self._prepare_static_features(df, id_to_idx)
        self.X = self._build_dataset_to_numpy_tensor(df, id_to_idx).permute(2, 0 ,1)
        #self.X_filled = self._build_dataset_to_numpy_tensor(df_filled, id_to_idx).permute(2, 0, 1)
        self.mask = is_finite(self.X)

    def _load_speed_data(self, file_path):
        file_path = glob.glob(f'{file_path}/*snappy.parquet')
        pf = ParquetFile(file_path)
        df = pf.to_pandas()
        df = self._remove_non_existing_edges_from_df(df)
        edges = [tuple((int(x[0]), int(x[1]))) for x in df[['from_node', 'to_node']].values]
        return edges, df

    def _remove_non_existing_edges_from_df(self, df):
        for idx, row in df.iterrows():
            arc = (int(row['from_node']), int(row['to_node']))
            if not self.G.has_node(arc[0]) or not self.G.has_node(arc[1]) or not self.G.has_edge(arc[0], arc[1]):
                df.drop(idx, inplace=True)
        return df

    def _load_new_cluster(self):
        cluster_id = next(self.mapping)
        edges, df = self._load_speed_data(os.path.join(cfg['all_cluster_path'], f"cluster_id={cluster_id}"))
        self.adj, L = get_adj_from_subgraph(self.G, cluster_id, edges)
        self._load_cluster_data(df, L=L)

    def _prepare_static_features(self, df, id_to_idx):
        self.cat_features = [['primary', 'asphalt', 'MajorRoad']] * len(df)
        self.ord_features_at_t = torch.FloatTensor(len(df))

        for _, row in df.iterrows():
            arc = (int(row['from_node']), int(row['to_node']))
            feat, road = self._arc_features(arc)
            self.cat_features[id_to_idx[arc]] = feat
            self.ord_features_at_t[id_to_idx[arc]] = float(self.road_enc[road])
        self.ord_features_at_t = self.ord_features_at_t.unsqueeze(1)

    def _build_dataset_to_numpy_tensor(self, df=None, id_to_idx=None):
        """We extract features from speed (actual speed, whether speed is missing)
        and combine with static features.

        Args:
            from_ (int): from time step
            to (int): to time step
            df (Pandas.DataFrame): the dataframe contains (only) speed data
            id_to_idx (dict): dictionary of segment_id --> df.indices

        Returns:
             np.ndarray: dataset tensor of shape [num_time_steps, num_nodes, num_features]

        """

        total_timesteps = df.shape[1] - 2
        NUM_FEATS = 96 + 3#53 # TODO change NUM_FEATS
        dataset = torch.Tensor(total_timesteps, len(df), NUM_FEATS)
        for t in range(total_timesteps):
            speed_is_nan_feature = torch.Tensor([0. if x != x else 1. for x in df[str(t)]]).unsqueeze(1)
            speed_features_at_t = torch.from_numpy(df[str(t)].to_numpy()).unsqueeze(1).float()
            # cat_features_at_t = map(lambda x: x + (t % self.nb_buckets, ), self.cat_features)
            # cat_features_at_t = map(lambda x: self.features_encoder[x], cat_features_at_t)
            # cat_features_at_t = map(lambda x: one_hot(x, len(self.features_encoder)), cat_features_at_t)
            # cat_features_at_t = torch.stack(list(cat_features_at_t))
            idxs = map(lambda x: x + (t % self.nb_buckets, ), self.cat_features)
            idxs = map(lambda x: self.features_encoder[x], idxs)
            idxs = torch.LongTensor([[i, j] for i, j in enumerate(idxs)])
            cat_features_at_t = torch.zeros((len(self.cat_features), len(self.features_encoder)), dtype=torch.float)
            dataset[t] = torch.cat([speed_features_at_t,
                                    speed_is_nan_feature,
                                    self.ord_features_at_t,
                                    cat_features_at_t], axis=1)

        return dataset

    def _arc_features(self, arc):
        arc = self.G[arc[0]][arc[1]]
        try:
            max_speed_feature = float(arc['data'].metadata.get('maxspeed', '50'))
            max_speed_feature = min(max_speed_cats,
                                    key=lambda x: abs(x - max_speed_feature))
        except:
            max_speed_feature = 50.
        try:
            num_lanes_feature = int(arc['data'].metadata.get('lanes', '1'))
            num_lanes_feature = min(num_lanes_cats,
                                    key=lambda x: abs(x - num_lanes_feature))
        except:
            num_lanes_feature = 1

        return (arc['data'].roadClass.name, ), (max_speed_feature, num_lanes_feature)

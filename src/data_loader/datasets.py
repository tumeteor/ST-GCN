import glob
import math

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import torch
from fastparquet import ParquetFile


class DatasetBuilder:
    def __init__(self, g):
        self.enc = OneHotEncoder(handle_unknown='ignore', categories=[
            ['access_ramp', 'corridor', 'living_street', 'platform', 'primary',
             'residential', 'secondary', 'secondary_link', 'service',
             'tertiary', 'tertiary_link', 'unclassified'], ['asphalt', 'cobblestone',
                                                            'cobblestone:flattened', 'concrete',
                                                            'concrete:plates', 'grass_paver', 'no_sur',
                                                            'paved',
                                                            'paving_stones', 'sett'],
            ['DirtRoad', 'LocalRoad', 'MajorRoad'],
            range(0, 24)
        ]
                                 )
        self.ienc = OrdinalEncoder(
            categories=[[5., 10., 20., 30., 40., 50., 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
                        [1., 2., 3., 4., 5.]])

        self.g = g

    def _arc_features(self, arc, timestep):
        arc = self.g[arc[0]][arc[1]]
        return [
                   arc['data'].metadata['highway'],
                   arc['data'].metadata.get('surface', 'no_sur'),
                   arc['data'].roadClass.name,
                   timestep % 24
               ], [float(arc['data'].metadata.get('maxspeed', '50')),
                   int(arc['data'].metadata.get('lanes', '1'))]

    def _construct_features(self, L):
        data = list()
        data_ord = list()
        for node in L.nodes:
            data.append(self._arc_features(node)[0])
            data_ord.append(self._arc_features(node)[1])
        return self.enc.fit_transform(data), self.ienc.fit_transform(data_ord)

    def _build_dataset_to_numpy_tensor(self, from_, to, df=None, id_to_idx=None):
        """
        We extract features from speed (actual speed, whether speed is missing)
        and combine with static features.
        :return:
             np.ndarray: dataset tensor of shape [num_time_steps, num_nodes, num_features]
        """
        dataset = list()
        for t in range(from_, to):
            cat_features_at_t = [['primary', 'asphalt', 'MajorRoad', t % 24]] * len(df)
            ord_features_at_t = [[50.0, 4]] * len(df)
            speed_features_at_t = [50] * len(df)
            speed_is_nan_feature = [1] * len(df)
            for _, row in df.iterrows():
                arc = (int(row['from_node']), int(row['to_node']))
                cat_features_at_t[id_to_idx[arc]], ord_features_at_t[id_to_idx[arc]] = \
                    self._arc_features(arc, timestep=t)
                speed_features_at_t[id_to_idx[arc]] = row[str(t)]
                if np.isnan(row[str(t)]):
                    speed_is_nan_feature[id_to_idx[arc]] = 0
            dataset.append(np.concatenate([np.array(speed_features_at_t).reshape(-1, 1),
                                           np.array(speed_is_nan_feature).reshape(-1, 1),
                                           self.ienc.fit_transform(ord_features_at_t),
                                           self.enc.fit_transform(cat_features_at_t).toarray()], axis=1))
        return np.stack(dataset, axis=0)

    def _generate_dataset_concat(self, X, X_masked, num_timesteps_input, num_timesteps_output):
        """
        Takes node features for the graph and divides them into multiple samples
        along the time-axis by sliding a window of size (num_timesteps_input+
        num_timesteps_output) across it in steps of 1.
        :param X: Node features of shape (num_vertices, num_features,
        num_timesteps)
        :return:
            - Node data (features + labels) divided into multiple samples. Shape is
              (num_samples, num_vertices, num_features, num_timesteps_input).

        """
        # Generate the beginning index and the ending index of a sample, which
        # contains (num_points_for_training + num_points_for_predicting) points
        indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                   in range(X.shape[2] - (
                    num_timesteps_input + num_timesteps_output) + 1)]

        # Save samples
        features, target = [], []
        mask = []
        for i, j in indices:
            # num_vertices, num_features, num_timesteps
            features.append(X[:, :, i: i + num_timesteps_input])
            target.append(X[:, 0, i + num_timesteps_input: j])
            mask.append(X_masked[:, 0, i + num_timesteps_input: j])

        return np.array(features), np.array(target), torch.stack(mask).numpy()

    def train_validation_test_split(self, X, X_filled, X_masked, look_back=29, look_ahead=1, split_ratio=[0.7, 0.9]):
        # num_vertices, num_features, num_timesteps
        split_line1 = int(X.shape[2] * split_ratio[0])
        split_line2 = int(X.shape[2] * split_ratio[1])
        train_original_data = X_filled[:, :, :split_line1]
        val_original_data = X_filled[:, :, split_line1:split_line2]
        test_original_data = X_filled[:, :, split_line2:]

        train_mask = X_masked[:, :, :split_line1]
        valid_mask = X_masked[:, :, split_line1:split_line2]
        test_mask = X_masked[:, :, split_line2:]

        # num_samples, num_nodes, num_timesteps, num_features
        training_data, training_target, train_mask = self._generate_dataset_concat(train_original_data, train_mask,
                                                                                   num_timesteps_input=look_back,
                                                                                   num_timesteps_output=look_ahead)
        valid_data, valid_target, valid_mask = self._generate_dataset_concat(val_original_data, valid_mask,
                                                                             num_timesteps_input=look_back,
                                                                             num_timesteps_output=look_ahead)
        test_data, test_target, test_mask = self._generate_dataset_concat(test_original_data, test_mask,
                                                                          num_timesteps_input=look_back,
                                                                          num_timesteps_output=look_ahead)

        data = {'train': training_data, 'valid': valid_data, 'test': test_data}
        target = {'train': training_target, 'valid': valid_target, 'test': test_target}
        mask = {'train': train_mask, 'valid': valid_mask, 'test': test_mask}
        return data, target, mask

    def fit_scaler(self, df_filled):
        speed_features = df_filled.values.flatten()
        speed_features = np.array([s for s in speed_features if not math.isnan(s)]).reshape(-1, 1)
        self.scaler.fit(speed_features)

    def _remove_non_existing_edges_from_df(self, df):
        for idx, row in df.iterrows():
            arc = (int(row['from_node']), int(row['to_node']))
            if not self.g.has_edge(arc[0], arc[1]) or not self.g.has_node(arc[0]) or not self.g.has_node(arc[1]):
                df.drop(idx, inplace=True)
        return df

    def load_speed_data(self, file_path):
        file_path = glob.glob(f'{file_path}/*snappy.parquet')
        pf = ParquetFile(file_path)
        df = pf.to_pandas()
        df = self._remove_non_existing_edges_from_df(df)
        edges = [tuple((int(x[0]), int(x[1]))) for x in df[['from_node', 'to_node']].values]
        return edges, df

    @staticmethod
    def _get_masks(X):
        # Build mask tensor
        X_masked = torch.where(torch.isnan(torch.from_numpy(X)), torch.tensor([0]), torch.tensor([1]))
        X_masked = X_masked.bool()
        return X_masked

    def construct_batches(self, df, L, TOTAL_T_STEPS=2263):
        id_to_idx = {}

        for idx, id_ in enumerate(L.nodes()):
            id_to_idx[id_] = idx

        df_filled = df.loc[:, df.columns != 'from_node']
        df_filled = df_filled.loc[:, df_filled.columns != 'to_node']

        # df_filled = df_filled.interpolate(method='nearest', axis=1)
        df_filled = df_filled.fillna(df_filled.mean())
        df_filled = df_filled.fillna(13.8)

        SPEED_COLUMNS = list(map(str, range(TOTAL_T_STEPS)))
        df.columns = ['from_node', 'to_node'] + SPEED_COLUMNS
        df_filled.columns = SPEED_COLUMNS
        df_filled['from_node'] = df['from_node']
        df_filled['to_node'] = df['to_node']

        df_filled = df_filled[['from_node', 'to_node'] + SPEED_COLUMNS[TOTAL_T_STEPS - 400:]]
        df = df[['from_node', 'to_node'] + SPEED_COLUMNS[TOTAL_T_STEPS - 400:]]

        X = self._build_dataset_to_numpy_tensor(df=df,
                                                id_to_idx=id_to_idx,
                                                from_=TOTAL_T_STEPS-400,
                                                to=TOTAL_T_STEPS)
        X_filled = self._build_dataset_to_numpy_tensor(df=df_filled,
                                                       id_to_idx=id_to_idx,
                                                       from_=TOTAL_T_STEPS-400,
                                                       to=TOTAL_T_STEPS)

        X = np.moveaxis(X, source=(0, 1, 2), destination=(2, 0, 1))
        X_filled = np.moveaxis(X_filled, source=(0, 1, 2), destination=(2, 0, 1))
        # Build mask tensor
        X_masked = self._get_masks(X)

        data, target, mask = self.train_validation_test_split(X=X, X_filled=X_filled, X_masked=X_masked)

        return data, target, mask

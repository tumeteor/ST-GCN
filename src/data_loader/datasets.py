from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import numpy as np
import torch

enc = OneHotEncoder(handle_unknown='ignore')
ienc = OrdinalEncoder()
scaler = StandardScaler()


def arc_features(arc, g):
    arc = g[arc[0]][arc[1]]
    return [
        arc['data'].metadata['highway'],
        arc['data'].metadata.get('surface', 'no_sur'),
        arc['data'].roadClass.name
    ],  [float(arc['data'].metadata.get('maxspeed', '50')),
        int(arc['data'].metadata.get('lanes', '1'))]


def construct_features(L):
    data = list()
    data_ord = list()
    for node in L.nodes:
        data.append(arc_features(node)[0])
        data_ord.append(arc_features(node)[1])
    return enc.fit_transform(data), ienc.fit_transform(data_ord)


def build_dataset_to_numpy_tensor(from_, to, df=None, norm=False, id_to_idx=None, L=None):
    """
    We extract features from speed (actual speed, whether speed is missing)
    and combine with static features.
    :return:
         np.ndarray: dataset tensor of shape [num_time_steps, num_nodes, num_features]
    """
    dataset = list()
    for t in range(from_, to):
        cat_features_at_t = [['primary', 'asphalt', 'MajorRoad']] * len(L.nodes)
        ord_features_at_t = [[50.0, 4]] * len(L.nodes)
        speed_features_at_t = [50] * len(L.nodes)
        speed_is_nan_feature = [1] * len(L.nodes)
        for _, row in df.iterrows():

            arc = (row['from_node'], row['to_node'])
            cat_features_at_t[id_to_idx[arc]], ord_features_at_t[id_to_idx[arc]]  = arc_features(arc)
            speed_features_at_t[id_to_idx[arc]] = row[str(t)]
            if np.isnan(row[str(t)]):
                speed_is_nan_feature[id_to_idx[arc]] = 0
        dataset.append(np.concatenate([scaler.transform(np.array(speed_features_at_t).reshape(-1, 1)) if norm else  np.array(speed_features_at_t).reshape(-1, 1),
                                       np.array(speed_is_nan_feature).reshape(-1, 1),
                                       ienc.fit_transform(ord_features_at_t),
                                       enc.fit_transform(cat_features_at_t).toarray()], axis=1))
    return np.stack(dataset, axis=0)


def generate_dataset_concat(X, X_masked, num_timesteps_input, num_timesteps_output):
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
        features.append(X[:, :, i: i + num_timesteps_input])
        target.append(X[:, 0, i + num_timesteps_input: j])
        mask.append(X_masked[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target)), torch.stack(mask)

import networkx as nx
import numpy as np
import argparse
import json
import logging

from logs import get_logger_settings, setup_logging
from src.config import Config
from src.reader import populate_graph_with_max_speed, read_jurbey_from_minio
from src.nmf.fast_nmf import train_nmf
from src.graph_utils import sample_graph_by_nodes


def nmf(X, K, iterations=100, alpha=0.0002, beta=0.02):
    """
    Simple matrix factorization using multiplicative update. The stochastic gradient descent approach
    only optimizes for non-zero values.
    https://www.ismll.uni-hildesheim.de/lehre/semML-16w/script/Group1_slides.pdf
    Args:
        X (scipy.sparse.csr_matrix): sparse matrix that contains mostly zeros
        K (int): number of latent components
        iterations: number of iters for optimization
        alpha: learning rate
        beta: parameter of regularization

    Returns:
      (numpy.ndarray, numpy.ndarray): 2D arrays of the decomposed matrices
    """
    m, n = X.shape
    W = np.random.rand(m, K)
    H = np.random.rand(n, K)
    H = H.T
    for _iter in range(iterations):
        ii, jj = X.nonzero()
        for i, j in zip(*(ii, jj)):
                if X[i, j] != np.nan:
                    eij = X[i, j] - np.dot(W[i, :], H[:, j])
                    for k in range(K):
                        # multiplicative update + L2 regularization
                        W[i][k] = W[i][k] + alpha * (2 * eij * H[k][j] - beta * W[i][k])
                        H[k][j] = H[k][j] + alpha * (2 * eij * W[i][k] - beta * H[k][j])
        tol = 0
        for i, j in zip(*(ii, jj)):
                if X[i, j] != np.nan:
                    # Frobenius norm
                    tol = tol + pow(X[i, j] - np.dot(W[i, :], H[:, j]), 2)
                    for k in range(K):
                        tol = tol + (beta / 2) * (pow(W[i][k], 2) + pow(H[k][j], 2))
        print(f'At iter: {_iter}, tol: {tol}')
        if tol < 0.001:
            break
    return W, H.T


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
    g = read_jurbey_from_minio(message['bucket'], message['jurbey_path'])
    logging.info("\u2705 Done loading Jurbey graph.")
    # sample to sub-graph
    g = sample_graph_by_nodes(g, 10000)
    # populate with max speed
    g = populate_graph_with_max_speed(g)
    # extract the adjacency matrix from graph
    A = nx.adjacency_matrix(g, weight="speed")
    from sklearn.preprocessing import MaxAbsScaler

    transformer = MaxAbsScaler().fit(A)
    A = transformer.transform(A)

    # node_list = list(g.nodes())
    # edge_list = list(g.edges())

    # W, H = nmf(X=A, K=10, iterations=10000)

    # df = nx.to_pandas_edgelist(g)
    # df = df[['source', 'target', 'speed']]
    # df.rename(columns={'speed': 'weight'})
    # df = df[0:1000]

    train_nmf(dataset=A)







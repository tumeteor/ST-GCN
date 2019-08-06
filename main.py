import random

import numpy as np
import argparse
import json
import logging
import networkx as nx
from logs import get_logger_settings, setup_logging
from src.config import Config
from src.reader import populate_graph_with_max_speed, read_jurbey_from_minio, populate_graph_with_fresh_speed, \
    get_dataframe_from_graph
from src.nmf.fast_nmf import train_nmf_with_dataframe, train_nmf_with_sparse_matrix
from src.graph_utils import sample_graph_by_nodes
from src.eval.measures import rmse


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
                if not np.isnan(X[i, j]):
                    eij = X[i, j] - np.dot(W[i, :], H[:, j])
                    for k in range(K):
                        # multiplicative update + L2 regularization
                        W[i][k] = W[i][k] + alpha * (2 * eij * H[k][j] - beta * W[i][k])
                        H[k][j] = H[k][j] + alpha * (2 * eij * W[i][k] - beta * H[k][j])
        tol = 0
        for i, j in zip(*(ii, jj)):
                if not np.isnan(X[i, j]):
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
    # populate with max speed
    g = populate_graph_with_max_speed(g)

    # sample to sub-graph
    g = sample_graph_by_nodes(g, 10000)

    # populate with fresh speed
    logging.info('\u2B07 Start populating speed to the graph...')
    g, fresh_edge_list = populate_graph_with_fresh_speed(g)
    print(f"number of fresh edges for the sampled graph: {len(fresh_edge_list)}")

    logging.info('\u2B07 Done populating speed to the graph...')

    # extract the adjacency matrix from graph
    A = nx.adjacency_matrix(g, weight="fresh_speed")
    B = nx.adjacency_matrix(g, weight="speed")

    # construct ground-truths
    gt_edges = random.sample(fresh_edge_list, 50)
    node_list = list(g.nodes())
    edge_list = list(g.edges())
    for u, v in gt_edges:
        g[u][v]["old_speed"] = A[node_list.index(u), node_list.index(v)]
        A[node_list.index(u), node_list.index(v)] = np.nan
        g[u][v]["speed"] = B[node_list.index(u), node_list.index(v)]
    W, H = nmf(X=A, K=10, iterations=2000)

    y_actual = list()
    y_pred = list()
    y_pred_max = list()
    for u, v in gt_edges:
        actual = g[u][v]["old_speed"]
        pred = np.dot(W[node_list.index(u)], H[node_list.index(v)])
        y_actual.append(actual)
        y_pred.append(pred)
        y_pred_max.append(g[u][v]["speed"])

    # do evaluation
    mf_rmse = rmse(y_actual, y_pred)
    print(f'rmse for MF: {rmse}')
    ms_rmse = rmse(y_actual, y_pred_max)
    print(f'rmse for max speed: {ms_rmse}')
    for x, y, z in zip(y_actual, y_pred, y_pred_max):
        print(x, y, z)

    # train_nmf_with_sparse_matrix(A)
    # df = get_dataframe_from_graph(g)
    # train_nmf_with_dataframe(df, len(df))









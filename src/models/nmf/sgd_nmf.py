import numpy as np


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

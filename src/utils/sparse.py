import torch.sparse
import numpy as np


def sparse_scipy2torch(matrix):
    """Convert a matrix from *any* `scipy.sparse` representation to a
    sparse `torch.tensor` value.
    """
    coo_matrix = matrix.tocoo()
    return torch.sparse.FloatTensor(
        torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col))),
        torch.FloatTensor(coo_matrix.data),
        torch.Size(coo_matrix.shape),
    )


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


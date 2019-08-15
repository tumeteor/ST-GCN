import torch.sparse
import numpy


def sparse_scipy2torch(matrix):
    """Convert a matrix from *any* `scipy.sparse` representation to a
    sparse `torch.tensor` value.
    """
    coo_matrix = matrix.tocoo()
    return torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((coo_matrix.row, coo_matrix.col))),
        torch.FloatTensor(coo_matrix.data),
        torch.Size(coo_matrix.shape),
    )

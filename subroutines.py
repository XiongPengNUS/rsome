import numpy as np
import scipy.sparse as sp
from numbers import Real
from scipy.sparse import csr_matrix


def index_array(shape):

    if isinstance(shape, tuple):
        size = np.prod(shape)
    else:
        size = int(shape)
        shape = int(shape),

    return np.arange(size, dtype=int).reshape(shape)


def sparse_array(shape):

    shape = shape if isinstance(shape, tuple) else (int(shape), )
    size = np.prod(shape).item()
    sparse = csr_matrix(([1.0]*size, np.arange(size, dtype=int),
                         np.arange(size+1, dtype=int)), (size, size))
    elements = [sparse[i, :] for i in range(size)]

    return np.array(elements).reshape(shape)


def array_to_sparse(array):

    size = array.size
    all_items = list(array.reshape((size, )))

    indices = np.concatenate(tuple(item.indices for item in all_items))
    data = np.concatenate(tuple([item.data for item in all_items]))
    indices = indices[data != 0]
    data = data[data != 0]

    indptr = np.zeros(size+1)
    for i in range(size):
        indptr[1+i] = indptr[i] + all_items[i].count_nonzero()

    return csr_matrix((data, indices, indptr), (size, all_items[0].shape[1]))


def sv_to_csr(array):

    size = array.size
    all_items = list(array.reshape((size, )))

    indices = np.concatenate(tuple(item.index for item in all_items))
    data = np.concatenate(tuple([item.value for item in all_items]))
    nonzero = (data != 0)
    indices = indices[nonzero]
    data = data[nonzero]

    indptr = np.zeros(size+1)
    for i in range(size):
        zero_counts = all_items[i].value.count(0)
        indptr[1+i] = indptr[i] + len(all_items[i].value) - zero_counts

    return csr_matrix((data, indices, indptr), (size, all_items[0].nvar))


def check_numeric(array):

    array = np.array(array.todense()) if sp.issparse(array) else array

    if isinstance(array, np.ndarray):
        if not isinstance(array.flat[0], np.number):
            raise TypeError('Incorrect data type of arrays.')
    elif not isinstance(array, Real):
        raise TypeError('Incorrect data type of arrays.')

    return array


def add_linear(left, right):

    if left.shape[1] > right.shape[1]:
        right.resize((right.shape[0], left.shape[1]))
    elif right.shape[1] > left.shape[1]:
        left.resize((left.shape[0], right.shape[1]))

    return left + right


def norm(affine, degree=2):

    return affine.norm(degree)


def square(affine):

    return affine.to_affine().square()


def sumsqr(affine):

    return affine.to_affine().sumsqr()

def E(expr):

    return expr.E

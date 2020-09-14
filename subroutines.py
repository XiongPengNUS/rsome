import numpy as np
import scipy.sparse as sp
from numbers import Real
from scipy.sparse import csr_matrix


def sparse_mul(ndarray, affine):

    array_ones = np.ones(ndarray.shape, dtype='int')
    affine_ind = np.arange(affine.size, dtype='int').reshape(affine.shape)
    affine_ones = np.ones(affine.shape, dtype='int')

    index = (array_ones * affine_ind).flatten()
    values = (ndarray * affine_ones).flatten()

    size = index.size

    return csr_matrix((values, (np.arange(size), index)),
                      shape=(size, affine.size))


def sp_matmul(ndarray, affine, shape):

    size = np.prod(shape)

    left = (ndarray.reshape((ndarray.size//ndarray.shape[-1],
                             ndarray.shape[-1])) if ndarray.ndim > 1
            else ndarray.reshape((1, ndarray.size)))
    right = (affine if len(affine.shape) > 1
             else affine.reshape((affine.size, 1)))

    row, inner = left.shape
    column = right.shape[-1]

    index = np.arange(affine.size).reshape(affine.shape)
    indim = index.ndim
    index = np.transpose(np.tile(index, (1, row)),
                         axes=list(range(indim-2)) + [indim-1,
                                                      indim-2]).flatten()
    index = np.tile(index, (size*inner//index.size, ))

    data = np.tile(ndarray, (1, column)).flatten()
    data = np.tile(data, size*inner//data.size)

    indptr = [inner*i for i in range(size+1)]

    return csr_matrix((data, index, indptr), shape=[size, affine.size])


def sp_lmatmul(ndarray, affine, shape):

    size = int(np.prod(shape))

    left = (affine.reshape((affine.size // affine.shape[-1],
                            affine.shape[-1])) if len(affine.shape) > 1
            else affine.reshape((1, affine.size)))
    right = (ndarray if ndarray.ndim > 1
             else ndarray.reshape((ndarray.size, 1)))

    row, inner = left.shape
    column = right.shape[-1]

    index = np.tile(np.arange(left.size).reshape(left.shape),
                    (1, column)).flatten()
    index = np.tile(index, (size * inner // index.size))
    andim = right.ndim
    data = np.tile(np.transpose(right,
                                axes=list(range(andim - 2)) + [andim - 1,
                                                               andim - 2]),
                   (size * inner // ndarray.size, 1)).flatten()
    indptr = [inner * i for i in range(size + 1)]

    return csr_matrix((data, index, indptr), shape=[size, affine.size])


def sp_trans(affine):

    index = np.arange(affine.size).reshape(affine.shape).T.flatten()
    indptr = np.arange(affine.size + 1)
    data = np.ones(affine.size)

    return csr_matrix((data, index, indptr), shape=[affine.size, affine.size])


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
    array = np.array([array]) if not isinstance(array, np.ndarray) else array

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


def event_dict(event_set):

    output = {}
    count = 0
    for item in event_set:
        for element in item:
            output[element] = count

        count += 1

    return output


def comb_set(s1, s2):

    d1 = event_dict(s1)
    d2 = event_dict(s2)

    dc = {item: str(d1[item]) + '-' + str(d2[item])
          for item in range(len(d1))}

    values = []
    output = []
    for key in dc:
        if dc[key] in values:
            index = values.index(dc[key])
            output[index].append(key)
        else:
            output.append([key])
            values.append(dc[key])

    return output


def norm(affine, degree=2):

    return affine.norm(degree)


def square(affine):

    return affine.to_affine().square()


def sumsqr(affine):

    return affine.to_affine().sumsqr()


def E(expr):

    return expr.E

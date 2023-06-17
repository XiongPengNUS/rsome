import numpy as np
import scipy.sparse as sp
from numbers import Real
from scipy.sparse import csr_matrix
from collections.abc import Iterable


def flat(a_list):
    flat_list = []
    for item in a_list:
        if isinstance(item, Iterable):
            flat_list.extend(flat(item))
        else:
            flat_list.append(item)

    return flat_list


def sparse_mul(ndarray, affine):

    array_ones = np.ones(ndarray.shape, dtype='int')
    affine_ind = np.arange(affine.size, dtype='int').reshape(affine.shape)
    affine_ones = np.ones(affine.shape, dtype='int')

    index = (array_ones * affine_ind).flatten()
    values = (ndarray * affine_ones).flatten()

    size = index.size

    return csr_matrix((values, (np.arange(size), index)),
                      shape=(size, affine.size))


def diag_comb(upper, lower):

    data = np.concatenate((upper.data, lower.data))
    indices = np.concatenate((upper.indices, upper.shape[1] + lower.indices))
    indptr = np.concatenate((upper.indptr[:-1], upper.indptr[-1] + lower.indptr))

    return csr_matrix((data, indices, indptr),
                      shape=np.array(upper.shape)+np.array(lower.shape))


def vert_comb(upper, lower):

    data = np.concatenate((upper.data, lower.data))
    indices = np.concatenate((upper.indices, lower.indices))
    indptr = np.concatenate((upper.indptr[:-1], upper.indptr[-1] + lower.indptr))

    shape = upper.shape[0] + lower.shape[0], max([upper.shape[1], lower.shape[1]])
    return csr_matrix((data, indices, indptr), shape=shape)


def sp_matmul(ndarray, affine, shape):

    size = int(np.prod(shape))

    if len(shape) < 1:
        return csr_matrix(ndarray)
    else:
        if len(affine.shape) == 1:
            affine = affine.reshape((affine.size, 1))
            row, col = shape[-1], 1
        elif len(ndarray.shape) == 1:
            ndarray = ndarray.reshape((1, ndarray.size))
            row, col = 1, shape[-1]
        else:
            row, col = shape[-2], shape[-1]

        inner = ndarray.shape[-1]

        affine_index = np.arange(affine.size).reshape(affine.shape)
        dim = len(affine.shape)
        axes = list(range(dim-2)) + [dim-1, dim-2]
        index = np.transpose(np.tile(affine_index, row), axes=axes).flatten()
        index_rep = size // (len(index)//inner)
        if index_rep > 1:
            index = np.tile(index, index_rep)

        data = np.tile(ndarray, col).flatten()
        data_rep = size // (len(data)//inner)
        if data_rep > 1:
            data = np.tile(data, data_rep)

        indptr = [inner*i for i in range(size+1)]

        return csr_matrix((data, index, indptr), shape=[size, affine.size])


def sp_lmatmul(ndarray, affine, shape):

    size = int(np.prod(shape))

    if len(shape) <= 0:
        return csr_matrix(ndarray)
    else:
        if len(affine.shape) == 1:
            affine = affine.reshape((1, affine.size))
            row, col = 1, shape[-1]
        elif len(ndarray.shape) == 1:
            ndarray = ndarray.reshape((ndarray.size, 1))
            row, col = shape[-1], 1
        else:
            row, col = shape[-2], shape[-1]

        inner = affine.shape[-1]

        affine_index = np.arange(affine.size).reshape(affine.shape)
        index = np.tile(affine_index, col).flatten()
        index_rep = size // (len(index)//inner)
        if index_rep > 1:
            index = np.tile(index, index_rep)

        dim = len(ndarray.shape)
        axes = list(range(dim-2)) + [dim-1, dim-2]
        data = np.transpose(np.tile(ndarray, row), axes=axes).flatten()
        data_rep = size // (len(data)//inner)
        if data_rep > 1:
            data = np.tile(data, data_rep)

        indptr = [inner*i for i in range(size+1)]

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

    if not isinstance(array, np.ndarray):
        size = 1
        all_items = [array]
    else:
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
    # array = np.array([array]) if not isinstance(array, np.ndarray) else array

    if isinstance(array, np.ndarray):
        if not isinstance(array.flat[0], np.number):
            raise TypeError('Incorrect data type of arrays.')
    elif not isinstance(array, Real):
        raise TypeError('Incorrect data type of arrays.')

    return array


def rso_broadcast(*args):

    arrays = [np.array(arg) if isinstance(arg, (Real, np.ndarray)) else arg.to_affine()
              for arg in args]

    indices = [np.arange(array.size).reshape(array.shape) for
               array in arrays]

    arrays = [array.reshape(array.size) for array in arrays]

    bdc = np.broadcast(*indices)
    outputs = [[item[i] for item, i in zip(arrays, index)] for index in bdc]

    return outputs


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


def E(expr):
    """
    The notion of the expectation of random variables and the worst-case
    expected values

    Notes
    -----
    This function is used to denote
    1. the expected value of an random variable when specifying the
    uncertainty set of expectations.
    2. the worst-case expected value of an RSOME expression involving random
    variables.
    """

    return expr.E

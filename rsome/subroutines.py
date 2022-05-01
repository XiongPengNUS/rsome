import numpy as np
import scipy.sparse as sp
from numbers import Real
from scipy.sparse import csr_matrix
from collections.abc import Iterable
from typing import List


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
            # outer = shape[:-1]
        else:
            row, col = shape[-2], shape[-1]
            # outer = shape[:-2]

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
    
    arrays = [np.array(arg) if isinstance(arg, Real) else arg
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

    values: List[str] = []
    output: List[List[int]] = []
    for key in dc:
        if dc[key] in values:
            index = values.index(dc[key])
            output[index].append(key)
        else:
            output.append([key])
            values.append(dc[key])

    return output


def norm(affine, degree=2):
    """
    Return the first, second, or infinity norm of a 1-D array.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. The array must be 1-D.
    degree : {1, 2, numpy.inf}, optional
        Order of the norm function. It can only be 1, 2, or infinity. The
        default value is 2, i.e., the Euclidean norm.

    Returns
    -------
    n : Convex
        The norm of the given array.
    """

    return affine.norm(degree)


def square(affine):
    """
    Return the element-wise square of an array.

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.

    Returns
    -------
    n : Convex
        The element-wise squares of the given array
    """

    return affine.to_affine().square()


def sumsqr(affine):
    """
    Return the sum of squares of an array.

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array. The array must be 1-D.

    Returns
    -------
    n : Convex
        The sum of squares of the given array
    """

    return affine.to_affine().sumsqr()


def quad(affine, qmat):
    """
    Return the quadratic expression affine @ qmat @ affine.

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array. The array must be 1-D.
    qmat : a positive or negative semidefinite matrix.

    Returns
    -------
    q : Convex
        The quadratic expression affine @ qmat affine
    """

    return affine.to_affine().quad(qmat)


def exp(affine):
    """
    Return the element-wise natural exponential function
    exp(affine).

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.

    Returns
    -------
    a : Convex
        The natural exponential function exp(affine)
    """

    return affine.exp()


def log(affine):
    """
    Return the element-wise natural logarithm function
    log(affine).

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.

    Returns
    -------
    a : Convex
        The natural logarithm function log(affine)
    """

    return affine.log()


def pexp(affine, scale):
    """
    Return the element-wise perspective of natural exponential 
    function scale * exp(affine/scale).

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.

    Returns
    -------
    a : PerspConvex
        The perspective of natural exponential function 
        scale * exp(affine/scale)
    """

    return affine.pexp(scale)


def plog(affine, scale):
    """
    Return the element-wise perspective of natural logarithm 
    function scale * log(affine/scale).

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.

    Returns
    -------
    a : PerspConvex
        The perspective of natural logarithm function
        scale * log(affine/scale)
    """

    return affine.plog(scale)


def entropy(affine):
    """
    Return the entropy expression sum(affine*log(affine)).

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array. It must be a vector.

    Returns
    -------
    a : Convex
        The entropy expression sum(affine*log(affine))
    """

    return affine.entropy()


def expcone(y, x, z):
    """
    Return an exponential cone constraint z*exp(x/z) <= y.

    Parameters
    ----------
    y : an array of variables or affine expression
        The right-hand-side of the constraint
    x : {Real, Vars, VarSub, Affine}
        The x value in the constraint. It must be a scalar.
    z : {Real, Vars, VarSub, Affine}
        The z value in the constraint. It must be a scalar.

    Returns
    -------
    constr : ExpConstr
        The exponential cone constraint z*exp(x/z) <= y
    """

    return y.expcone(x, z)


def kldiv(p, phat, r):
    """
    Return an KL divergence constraint sum(p*log(p/phat)) <= r.

    Parameters
    ----------
    p : an array of variables or affine expression
        The array of probabilities. It must be a vector.
    phat : {Real, Vars, VarSub, Affine}
        The array of empirical probabilities. It must a vector with
        the same shape as p.
    r : {Real, Vars, VarSub, Affine}
        The ambiguity constant. It must be a scalar.

    Returns
    -------
    constr : ExpConstr
        The KL divergence constraint sum(p*log(p/phat)) <= r.
    """

    return p.kldiv(phat, r)


def E(expr):
    """
    The notion of the expectation of random variables and the worst-case
    expected values

    Notes
    -----
    This function is used to denote
    1. the expected value of an random variable when specifying the
    uncertainty set of expectations.
    2. the worst-case expected value of an affine function involving random
    variables.
    """

    return expr.E

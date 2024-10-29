from .lp import PiecewiseConvex
from .lp import Vars, VarSub, Affine
from .lp import DecRule, DecRuleSub
from .lp import RoAffine
from .lp import concat
from .subroutines import *
from numbers import Real
import numpy as np


def norm(affine, degree=2, method=None):
    """
    Return the first, second, or infinity norm of a 1-D array.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. The array must be 1-D.
    degree : {1, 2, numpy.inf}, optional
        Order of the norm function. It can only be 1, 2, or infinity.
        The default value is 2, i.e., the Euclidean norm.

    Returns
    -------
    out : Convex
        The norm of the given array.
    """

    return affine.norm(degree, method)


def pnorm(affine, degree=2, method=None):
    """
    Return the p-norm of a 1-D array, where p is a real number larger
    than 1.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. The array must be 1-D.
    degree : int, float, numpy.inf
        Order of the norm function. It must be a real number larger
        than 1.
    method : {'soc', 'exc'}
        The reformulation method. 'soc' for reformulating the p-norm
        constraint into second-order conic constraints. 'exc' for
        reformulating the p-norm constraint into exponential conic
        constraints.

    Returns
    -------
    out : Convex
        The norm of the given array.
    """

    return affine.pnorm(degree, method)


def gmean(affine, beta=None):
    """
    Return the weighted geometric mean of a 1-D array. The weights
    are specified by an array-like structure beta. It is expressed
    as prod(affine ** beta) ** (1/sum(beta))

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array. The array must be 1-D.
    beta : None of an iterable of integers.
        The weight of each term in the geometric mean expression. Each
        weight must be an integer no smaller than one. When beta is None
        (by default), all weights are ones.

    Returns
    -------
    out : Convex
        A convex expression representing the geometric mean of the given
        array.
    """

    return affine.gmean(beta)


def fnorm(*args):
    """
    Return the Frobenius norm of all given arrays, regardless
    of their shapes.

    Parameters
    ----------
    arg : an array of variables or affine expression
        Input array.

    Returns
    -------
    out : Convex
        The Frobenius norm of all given arrays. These arrays do
        not have to be two-dimensional.
    """

    iters = []
    for arg in args:
        arg = arg.to_affine()
        iters.append(arg.reshape((arg.size, )))

    affine = concat(iters)

    return affine.to_affine().norm(2)


def square(affine):
    """
    Return the element-wise square of an array.

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.

    Returns
    -------
    out : Convex
        The element-wise squares of the given array
    """

    return affine.to_affine().square()


def power(affine, p, q=1):
    """
    Return the element-wise power of an array x, expressed
    as abs(affine) ** (p/q)

    Parameters
    ----------
    affine : an array of variables or affine expression
        Input array.
    p : an integer or an array of integers.
        Exponent values.
    q : an integer or an array of integers.
        Denominators of exponent values

    Returns
    -------
    out : Convex
        The element-wise powers of the given array
    """

    return affine.to_affine().power(p, q)


def sumsqr(*args):
    """
    Return the sum of squares of elements in all given arrays,
    regardless of their shapes.

    Parameters
    ----------
    arg : an array of variables or affine expression
        Input array.

    Returns
    -------
    out : Convex
        The sum of squares of elements of all input arrays.
    """

    iters = []
    for arg in args:
        arg = arg.to_affine()
        iters.append(arg.reshape((arg.size, )))

    affine = concat(iters)

    return affine.to_affine().sumsqr()


def quad(x, qmat):
    """
    Return the quadratic expression x @ qmat @ x.

    Parameters
    ----------
    x : an array of variables or affine expressions
        Input array. The array must be 1-D.
    qmat : a positive or negative semidefinite matrix.
        A matrix representing the quadratic coefficients.

    Returns
    -------
    out : Convex
        The quadratic expression affine @ qmat affine
    """

    return x.to_affine().quad(qmat)


def rsocone(x, y, z):
    """
    Return a rotated cone constraint sumsqr(x) <= y*z.

    Parameters
    ----------
    x : {Iterables, Vars, VarSub, Affine}
        Input arrays. If x is a collection of arrays, the
        left-hand-side expression is the sum of squares of
        all elements in given arrays.
    y : {Real, Vars, VarSub, Affine}
        The y value in the constraint. It must be a scalar.
    z : {Real, Vars, VarSub, Affine}
        The z value in the constraint. It must be a scalar.

    Returns
    -------
    output : CvxConstr
        The rotated cone constraint
    """

    if isinstance(x, Iterable):
        iters = []
        for arg in x:
            arg = arg.to_affine()
            iters.append(arg.reshape((arg.size, )))

        affine = concat(iters)
    else:
        affine = x

    return affine.to_affine().rsocone(y, z)


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
    out : Convex
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
    out : Convex
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

    scale : {Real, Vars, VarSub, Affine}
        The scale value in the perspective function. It must be a
        scalar.

    Returns
    -------
    out : PerspConvex
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

    scale : {Real, Vars, VarSub, Affine}
        The scale value in the perspective function. It must be a
        scalar.

    Returns
    -------
    out : PerspConvex
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
    out : Convex
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


def softplus(x):
    """
    Return the element-wise softplus function log(1 + exp(x)).

    Parameters
    ----------
    x : {Vars, VarSub, Affine}
        The x value as the function input.

    Returns
    -------
    constr : Convex
        The element-wise softplus function log(1 + exp(x)).
    """

    return x.softplus()


def kldiv(p, q, r):
    """
    Return an KL divergence constraint sum(p*log(p/phat)) <= r.

    Parameters
    ----------
    p : an array of variables or affine expression
        The array of probabilities. It must be a vector.
    q : {Real, Vars, VarSub, Affine}
        The array of empirical probabilities. It must a vector with
        the same shape as p.
    r : {Real, Vars, VarSub, Affine}
        The ambiguity constant. It must be a scalar.

    Returns
    -------
    constr : ExpConstr
        The KL divergence constraint sum(p*log(p/q)) <= r.
    """

    return p.kldiv(q, r)


def trace(affine):
    """
    Return the trace of a 2-D array.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. It must be a 2-D array.

    Returns
    -------
    out : Affine
        Output array as the trace of the 2-D array.
    """

    return affine.trace()


def maxof(*args):
    """
    Return a piecewise function of the maximum of a number of
    expressions.

    Parameters
    ----------
    *args : RSOME affine or bi-affine expressions, real numbers
        Expressions or numerical values used to define the piecewise
        function.

    Returns
    -------
    piecewise : PiecewiseConvex
        The piecewise function defined to be the maximum of a number
        of given expressions or numerical values.
    """

    pieces = flat(args)
    this_model = None
    for arg in pieces:
        if isinstance(arg, (Vars, VarSub, Affine)):
            arg = arg.to_affine()
            if this_model is None:
                this_model = arg.model.top
            elif arg.model.top is not this_model:
                raise ValueError('Models not match.')
        elif isinstance(arg, RoAffine):
            if this_model is None:
                this_model = arg.affine.model.top
        elif isinstance(arg, (DecRule, DecRuleSub)):
            arg = arg.to_affine()
            if this_model is None:
                this_model = arg.model.top
        elif isinstance(arg, Real):
            arg = np.array([arg])
        elif not isinstance(arg, np.ndarray):
            raise TypeError('Unsupported expressions.')

        if arg.size != 1:
            raise ValueError('All pieces must have their sizes to be one.')

    if this_model is None:
        raise ValueError('Incorrect piecewise function.')

    return PiecewiseConvex(this_model, pieces, sign=1, add_sign=1)


def minof(*args):
    """
    Return a piecewise function of the minimum of a number of
    expressions.

    Parameters
    ----------
    *args : RSOME affine or bi-affine expressions, real numbers
        Expressions or numerical values used to define the piecewise
        function.

    Returns
    -------
    piecewise : PiecewiseConvex
        The piecewise function defined to be the minimum of a number
        of given expressions or numerical values.
    """

    piecewise = maxof(*args)
    piecewise.pieces = [-piece for piece in piecewise.pieces]

    piecewise.sign = -1
    piecewise.add_sign = -1

    return piecewise


def diag(affine, k=0, fill=False):
    """
    Return the diagonal elements of a 2-D array.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. It must be a 2-D array.

    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    fill : bool
        If True, return a 2-D array where the non-diagonal elements are
        filled with zeros. Otherwise, return the diagonal elements as a
        1-D array

    Returns
    -------
    out : Affine
        The diagonal elements of a given 2-D array.
    """

    return affine.diag(k, fill)


def tril(affine, k=0):
    """
    Return the lower triangular elements of a 2-D array. The remaining
    elements are filled with zeros.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. It must be a 2-D array.

    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is
        the main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    out : Affine
        The lower triangular elements of the given 2-D array.
    """

    return affine.tril(k)


def triu(affine, k):
    """
    Return the upper triangular elements of a 2-D array. The remaining
    elements are filled with zeros.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. It must be a 2-D array.

    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default)
        is the main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    out : Affine
        The upper triangular elements of the given 2-D array.
    """

    return affine.triu(k)


def logdet(affine):
    """
    Return the log-determinant of a positive semidefinite matrix
    expressed as a two-dimensional array.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. It must be a 2-D array.

    Returns
    -------
    out : Affine
        The scalar representing the log-determinant of the given
        two-dimensional array.
    """

    return affine.to_affine().logdet()


def rootdet(affine):
    """
    Return the root-determinant of a positive semidefinite matrix
    expressed as a two-dimensional array. The root-determinant is
    expressed as (det(A))**(1/L), where L is the dimension of the
    two-dimensinoal array.

    Parameters
    ----------
    affine : an array of variables or affine expressions.
        Input array. It must be a 2-D array.

    Returns
    -------
    out : Affine
        The scalar representing the root-determinant of the given
        two-dimensional array.
    """

    return affine.to_affine().rootdet()

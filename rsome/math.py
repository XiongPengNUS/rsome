from .lp import PiecewiseConvex
from .lp import Vars, VarSub, Affine
from .lp import DecRule, DecRuleSub
from .lp import RoAffine
from .subroutines import *
from numbers import Real
import numpy as np


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

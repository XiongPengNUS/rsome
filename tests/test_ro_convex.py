import rsome as rso
from rsome import ro
from rsome import grb_solver as grb
from rsome import eco_solver as eco
import numpy as np
import numpy.random as rd
import pytest


@pytest.mark.parametrize('array, const', [
    (rd.rand(8), rd.rand()),
    (rd.rand(8), rd.rand(5)),
    (rd.rand(1), rd.rand(1))
])
def test_norm_one(array, const):

    target = abs(array).sum() + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = rso.norm(x, 1) + const
    m.min(a)
    m.st(a >= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.max()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'a one-norm expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} one-norm expression{suffix}'

    target = -2*abs(array).sum() + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr1, expr2 = rso.norm(x, 1), x.norm(1)
    assert (expr1.affine_in.linear.toarray() ==
            expr2.affine_in.linear.toarray()).all()
    assert (expr1.affine_in.const == expr2.affine_in.const).all()
    assert (expr1.affine_out == expr2.affine_out).all()

    expr = (-2)*rso.norm(x, 1) + const
    m.max(a)
    m.st(a <= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.min()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'a one-norm expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} one-norm expression{suffix}'

    with pytest.raises(ValueError):
        rso.norm(x, 1) + const >= 0

    with pytest.raises(ValueError):
        -rso.norm(x, 1) + const <= 0

    with pytest.raises(TypeError):
        rso.norm(x, 1) + const == 0

    with pytest.raises(ValueError):
        rso.norm(x, 3) + const == 0

    with pytest.raises(ValueError):
        rso.norm(x, '2') + const == 0

    constr = (a <= expr)
    cnum = constr.affine_out.size
    suffix = '' if cnum == 1 else 's'
    assert constr.__repr__() == '{} convex constraint{}'.format(cnum, suffix)


@pytest.mark.parametrize('array, const', [
    (rd.rand(8), rd.rand()),
    (rd.rand(8), rd.rand(5)),
    (rd.rand(1), rd.rand(1))
])
def test_norm_two(array, const):

    target = (array**2).sum()**0.5 + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr1, expr2 = rso.norm(x, 2), x.norm(2)
    assert (expr1.affine_in.linear.toarray() ==
            expr2.affine_in.linear.toarray()).all()
    assert (expr1.affine_in.const == expr2.affine_in.const).all()
    assert (expr1.affine_out == expr2.affine_out).all()

    expr = rso.norm(x, 2) + const
    m.min(a)
    m.st(a >= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.max()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'an Eclidean norm expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} Eclidean norm expression{suffix}'

    target = -(array**2).sum()**0.5 + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = - rso.norm(x, 2) + const
    m.max(a)
    m.st(a <= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.min()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'an Eclidean norm expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} Eclidean norm expression{suffix}'

    with pytest.raises(ValueError):
        rso.norm(x, 2) + const >= 0

    with pytest.raises(ValueError):
        -rso.norm(x, 2) + const <= 0

    with pytest.raises(TypeError):
        rso.norm(x, 2) + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(8), rd.rand()),
    (rd.rand(8), rd.rand(5)),
    (rd.rand(1), rd.rand(1))
])
def test_norm_inf(array, const):

    target = array.max() + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr1, expr2 = rso.norm(x, 'inf'), x.norm('inf')
    assert (expr1.affine_in.linear.toarray() ==
            expr2.affine_in.linear.toarray()).all()
    assert (expr1.affine_in.const == expr2.affine_in.const).all()
    assert (expr1.affine_out == expr2.affine_out).all()

    expr = rso.norm(x, 'inf') + const
    m.min(a)
    m.st(a >= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.max()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'an infinity norm expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} infinity norm expression{suffix}'

    target = -array.max() + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = - rso.norm(x, 'inf') + const
    m.max(a)
    m.st(a <= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.min()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'an infinity norm expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} infinity norm expression{suffix}'

    with pytest.raises(ValueError):
        rso.norm(x, 'inf') + const >= 0

    with pytest.raises(ValueError):
        -rso.norm(x, 'inf') + const <= 0

    with pytest.raises(TypeError):
        rso.norm(x, 'inf') + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(8), rd.rand()),
    (rd.rand(8), rd.rand(8)),
    (rd.rand(1), rd.rand(1)),
    (rd.rand(8), rd.rand(3, 8)),
    (rd.rand(3, 5), rd.rand(3, 1)),
    (rd.rand(3, 5), rd.rand(5)),
    (rd.rand(3, 5), rd.rand(2, 3, 5)),
    (rd.rand(2, 3, 5), rd.rand(3, 5)),
    (rd.rand(2, 3, 3, 2), rd.rand(3, 1)),
    (rd.rand(2, 1, 2, 3), rd.rand(2, 3)),
    (rd.rand(2, 3), rd.rand(2, 1, 2, 3))
])
def test_squares(array, const):

    target = array**2 + const

    m = ro.Model()
    a = m.dvar()
    d = m.dvar(target.shape)
    x = m.dvar(array.shape)

    expr1, expr2 = rso.square(x), x.square()
    assert (expr1.affine_in.linear.toarray() ==
            expr2.affine_in.linear.toarray()).all()
    assert (expr1.affine_in.const == expr2.affine_in.const).all()
    assert (expr1.affine_out == expr2.affine_out).all()

    expr = rso.square(x) + const
    m.min(a)
    m.st([a >= d, d >= expr - target])
    # m.st(d >= expr - target)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'an element-wise square expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        repr = f'{shape_str} element-wise square expression{suffix}'
        assert expr.__repr__() == repr

    target = -(array**2) + const

    m = ro.Model()
    a = m.dvar()
    d = m.dvar(target.shape)
    x = m.dvar(array.shape)

    expr = - rso.square(x) + const
    m.max(a)
    # m.st(a <= d)
    m.st(a <= d, d <= expr - target)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'an element-wise square expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        repr = f'{shape_str} element-wise square expression{suffix}'
        assert expr.__repr__() == repr

    with pytest.raises(ValueError):
        rso.square(x) + const >= 0

    with pytest.raises(ValueError):
        -rso.square(x) + const <= 0

    with pytest.raises(TypeError):
        rso.square(x) + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(8), rd.rand()),
    (rd.rand(8), rd.rand(5)),
    (rd.rand(1), rd.rand(1))
])
def test_square_sum(array, const):

    target = (array**2).sum() + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr1, expr2 = rso.sumsqr(x), x.sumsqr()
    assert (expr1.affine_in.linear.toarray() ==
            expr2.affine_in.linear.toarray()).all()
    assert (expr1.affine_in.const == expr2.affine_in.const).all()
    assert (expr1.affine_out == expr2.affine_out).all()

    expr = rso.sumsqr(x) + const
    m.min(a)
    m.st(a >= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.max()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'a sum of squares expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} sum of squares expression{suffix}'

    target = -(array**2).sum() + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = - rso.sumsqr(x) + const
    m.max(a)
    m.st(a <= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.min()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'a sum of squares expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} sum of squares expression{suffix}'

    with pytest.raises(ValueError):
        rso.sumsqr(x) + const >= 0

    with pytest.raises(ValueError):
        -rso.sumsqr(x) + const <= 0

    with pytest.raises(TypeError):
        rso.sumsqr(x) + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(5), rd.rand()),
    (rd.rand(5), rd.rand(3)),
    (rd.rand(1), rd.rand(1))
])
def test_quad(array, const):

    vec = rd.rand(array.shape[0])
    qmat = np.diag(vec)

    target = array@qmat@array + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = const + x.quad(qmat)
    m.min(a)
    m.st(a >= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.max()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'a sum of squares expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} sum of squares expression{suffix}'

    target = -array@qmat@array + const

    m = ro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = - x.quad(qmat) + const
    m.max(a)
    m.st(a <= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.min()) < 1e-4
    assert type(expr) == ro.Convex
    if target.shape == ():
        assert expr.__repr__() == 'a sum of squares expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} sum of squares expression{suffix}'

    with pytest.raises(ValueError):
        x.quad(qmat) + const >= 0

    with pytest.raises(ValueError):
        -x.quad(qmat) + const <= 0

    with pytest.raises(TypeError):
        rso.quad(x, qmat) + const == 0


rd.seed(1)


@pytest.mark.parametrize('xvalue, zvalue', [
    (rd.rand(), rd.rand()),
    (rd.rand(), rd.rand())
])
def test_expcone(xvalue, zvalue):

    m = ro.Model()
    x = m.dvar()
    y = m.dvar()
    z = m.dvar()

    m.min(y)
    m.st(x == xvalue)
    m.st(z == zvalue)
    m.st(rso.expcone(y, x, z))

    m.solve(eco)
    target = zvalue*np.exp(xvalue/zvalue)

    assert abs(m.get() - target.min()) < 1e-4

    m = ro.Model()
    y = m.dvar()
    z = m.dvar()

    m.min(y)
    m.st(z == zvalue)
    m.st(rso.expcone(y, 3*xvalue + 2, 2.5*z + 1.2))

    m.solve(eco)
    target = (2.5*zvalue+1.2) * np.exp((3*xvalue+2)/(2.5*zvalue+1.2))

    assert abs(m.get() - target.min()) < 1e-4

    m = ro.Model()
    x = m.dvar(3)
    y = m.dvar(3)
    z = m.dvar()

    m.min(y.sum())
    m.st(x == xvalue)
    m.st(z == zvalue)
    m.st(rso.expcone(y, 1.2*xvalue + 2, 1.5*z + 1.2))

    m.solve(eco)
    target = 3*(1.5*zvalue+1.2) * np.exp((1.2*xvalue+2)/(1.5*zvalue+1.2))

    assert abs(m.get() - target.min()) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    with pytest.raises(ValueError):
        rso.expcone(y, np.ones(3), z)

    with pytest.raises(ValueError):
        rso.expcone(y, x, z)

    with pytest.raises(ValueError):
        rso.expcone(y, z, x)


rd.seed(1)


@pytest.mark.parametrize('xvalue', [
    rd.rand(3, 5),
    rd.rand(2, 3, 2)
])
def test_exp(xvalue):

    shape = xvalue.shape
    m = ro.Model()
    x = m.dvar(shape)
    y = m.dvar(shape)

    m.min(y.sum())
    m.st(x == xvalue)
    m.st(3*y + 2.3 >= rso.exp(2.1*x + 1.5))

    m.solve(eco)
    target = ((np.exp(2.1*xvalue + 1.5) - 2.3) / 3).sum()

    assert abs(m.get() - target) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    m = ro.Model()
    x = m.dvar(shape)
    y = m.dvar(shape)

    m.min(y.sum())
    m.st(x == xvalue)
    m.st(y >= rso.exp(2.1*x + 1.5))

    m.solve(eco)
    target = np.exp(2.1*xvalue + 1.5).sum()

    assert abs(m.get() - target) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    with pytest.raises(ValueError):
        rso.exp(x) >= y


rd.seed(2)


@pytest.mark.parametrize('xvalue, scales', [
    (rd.rand(3, 5), np.maximum(0.2, rd.rand(3, 5))),
    (rd.rand(2, 3, 2), rd.rand(3, 2))
])
def test_pexp(xvalue, scales):

    shape1 = xvalue.shape
    targets = ((scales*np.exp(2.1*xvalue/scales) - 2.3) / 3)
    shape = targets.shape

    m = ro.Model()
    x = m.dvar(shape1)
    y = m.dvar(shape)

    m.min(y.sum())
    m.st(3*y + 2.3 >= rso.pexp(2.1*x, scales))
    m.st(x == xvalue)
    m.solve(eco)

    assert abs(m.get() - targets.sum()) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    with pytest.raises(ValueError):
        rso.pexp(x, 2.5) >= y


rd.seed(1)


@pytest.mark.parametrize('xvalue', [
    rd.rand(3, 5),
    rd.rand(2, 3, 2)
])
def test_log(xvalue):

    shape = xvalue.shape
    m = ro.Model()
    x = m.dvar(shape)
    y = m.dvar(shape)

    m.max(y.sum())
    m.st(y <= rso.log(2.1*x + 1.5))
    m.st(x == xvalue)

    m.solve(eco)
    target = np.log(2.1*xvalue + 1.5).sum()

    assert abs(m.get() - target) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    m = ro.Model()
    x = m.dvar(shape)
    y = m.dvar(shape)

    m.max(y.sum())
    m.st(x == xvalue)
    m.st(y <= rso.log(2.1*x + 1.5))

    m.solve(eco)
    target = np.log(2.1*xvalue + 1.5).sum()

    assert abs(m.get() - target) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    with pytest.raises(ValueError):
        rso.log(x) <= y


rd.seed(2)


@pytest.mark.parametrize('xvalue, scales', [
    (rd.rand(3, 5), np.maximum(0.2, rd.rand(3, 5))),
    (rd.rand(2, 3, 2), rd.rand(3, 2))
])
def test_plog(xvalue, scales):

    shape1 = xvalue.shape
    targets = ((scales*np.log(2.1*xvalue/scales) - 2.3) / 3)
    shape = targets.shape

    m = ro.Model()
    x = m.dvar(shape1)
    y = m.dvar(shape)

    m.max(y.sum())
    m.st(3*y + 2.3 <= rso.plog(2.1*x, scales))
    m.st(x == xvalue)
    m.solve(eco)

    assert abs(m.get() - targets.sum()) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    with pytest.raises(ValueError):
        rso.plog(x, 1.5) <= y


@pytest.mark.parametrize('xvalue', [
    rd.rand(3),
    rd.rand(6)
])
def test_entropy(xvalue):

    shape = xvalue.shape
    m = ro.Model()
    x = m.dvar(shape)
    y = m.dvar(shape + (2,))

    m.max(3.5 * rso.entropy(0.5*x + 0.6))
    m.st(x == xvalue)

    m.solve(eco)
    target = - 3.5 * ((0.5*xvalue + 0.6) * np.log(0.5*xvalue + 0.6)).sum()

    assert abs(m.get() - target) < 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    with pytest.raises(ValueError):
        rso.entropy(x) <= y

    with pytest.raises(ValueError):
        rso.entropy(y) >= x[0]


@pytest.mark.parametrize('array, r', [
    (rd.rand(3), 0.02),
    (rd.rand(5), 0.01),
    (rd.rand(9), 0.001),
])
def test_kl_divergence(array, r):

    phat = array / array.sum()
    ns = len(phat)

    m = ro.Model()
    p = m.dvar(ns)

    c = np.random.rand(ns)
    m.max(c @ p)
    m.st(p.kldiv(phat, r))
    m.st(p >= 0, p.sum() == 1)
    m.solve(eco)

    ps = p.get()

    div = (ps * np.log(ps/phat)).sum()

    assert abs(r - div) <= 1e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


def test_convex_err():

    m1 = ro.Model()
    x = m1.dvar(5)
    xx = m1.dvar((6, 7))

    with pytest.raises(ValueError):
        rso.norm(xx, 1)

    with pytest.raises(ValueError):
        rso.norm(xx, 2)

    with pytest.raises(ValueError):
        rso.norm(xx, 'inf')

    with pytest.raises(ValueError):
        rso.sumsqr(xx)

    with pytest.raises(TypeError):
        rso.sumsqr(x) + rso.norm(x, 2)

    with pytest.raises(TypeError):
        rso.square(x) * rd.rand(5)

    """
    with pytest.raises(ValueError):
        rso.expcone(x, x, xx[0, 0])

    with pytest.raises(ValueError):
        rso.expcone(x, xx[0, 0], x)

    with pytest.raises(ValueError):
        rso.kldiv(x, xx, 0.05)

    with pytest.raises(ValueError):
        rso.kldiv(x, xx[0], 0.05)
    """

    vec = rd.rand(7)
    qmat = vec.reshape((vec.size, 1)) @ vec.reshape((1, vec.size))
    with pytest.raises(ValueError):
        rso.quad(xx, qmat)

    qmat = rd.rand(5, 5)
    with pytest.raises(ValueError):
        rso.quad(x, qmat)

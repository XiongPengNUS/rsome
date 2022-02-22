import rsome as rso
from rsome import ro
from rsome import grb_solver as grb
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
    (rd.rand(2, 4), rd.rand(2, 3, 2, 4))
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


def test_convex_err():

    model = ro.Model()
    x = model.dvar(5)
    xx = model.dvar((6, 7))

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

    vec = rd.rand(7)
    qmat = vec.reshape((vec.size, 1)) @ vec.reshape((1, vec.size))
    with pytest.raises(ValueError):
        rso.quad(xx, qmat)

    qmat = rd.rand(5, 5)
    with pytest.raises(ValueError):
        rso.quad(x, qmat)

from rsome import dro
from rsome import E
import numpy as np
import pytest


def test_dvar_scalar():

    model = dro.Model(5)

    x = model.dvar(name='x')
    y = model.dvar(name='y')
    b = model.dvar(name='b', vtype='b')
    z = model.rvar(name='z')

    assert x.shape == ()
    assert x.ndim == 0
    assert x.__repr__() == 'x: a static decision variable'
    assert y.shape == ()
    assert y.ndim == 0
    assert y.__repr__() == 'y: a static decision variable'
    assert z.shape == ()
    assert z.ndim == 0
    assert z.__repr__() == 'z: a random variable'
    expect_z = E(z)
    assert expect_z.shape == ()
    assert expect_z.ndim == 0
    assert expect_z.__repr__() == 'E(z): an expectation of random variable'

    y.adapt(2)
    assert y.__repr__() == 'y: an event-wise static decision variable'
    ya = y.to_affine()
    assert ya.shape == ()
    assert ya.const == 0
    assert ya.__repr__() == 'an event-wise affine expression'
    assert (ya.linear.toarray() == np.array([0, 0, 1, 0])).all()
    assert ya.const == 0
    assert ya.event_adapt == [[0, 1, 3, 4], [2]]

    x.adapt(z)
    assert x.__repr__() == 'x: an affinely adaptive decision variable'
    xa = x.to_affine()
    assert xa.shape == ()
    assert xa.const == 0
    assert xa.__repr__() == 'a bi-affine expression'
    assert (xa.linear.toarray() == np.array([0, 1, 0, 0])).all()
    assert xa.const == 0
    assert xa.event_adapt == [[0, 1, 2, 3, 4]]

    x.adapt(3)
    assert x.__repr__() == 'x: an event-wise affinely adaptive decision variable'
    xa = x.to_affine()
    assert xa.shape == ()
    assert xa.const == 0
    assert xa.__repr__() == 'an event-wise bi-affine expression'
    assert (xa.linear.toarray() == np.array([0, 1, 0, 0])).all()
    assert xa.const == 0
    assert xa.event_adapt == [[0, 1, 2, 4], [3]]

    with pytest.raises(ValueError):
        model.dvar(vtype='d')

    with pytest.raises(ValueError):
        model.dvar(vtype='BI')

    with pytest.raises(ValueError):
        b.adapt(z)

    with pytest.raises(TypeError):
        x.adapt(y)

    with pytest.raises(KeyError):
        x.adapt(10)

    with pytest.raises(KeyError):
        x.adapt(range(3, 5))


@pytest.mark.parametrize('length, shape', [
    (5, (3, 4)),
    (6, (5, 2)),
    (3, (3, 4, 5))
])
def test_dvar_array(length, shape):

    model = dro.Model(10)
    x = model.dvar(length, name='x')
    y = model.dvar(shape, vtype='b', name='y')
    z = model.rvar(3)

    assert x.shape == (length, )
    assert x.ndim == 1
    assert x.vtype == 'C'
    assert x.__repr__() == f'x: {length} static decision variables'

    assert y.shape == shape
    assert y.ndim == len(shape)
    assert y.vtype == 'B'
    shape_str = 'x'.join([str(dim) for dim in shape])
    assert y.__repr__() == f'y: {shape_str} static decision variables'

    y.adapt(3)
    y.adapt([2, 5, 6])
    assert y.__repr__() == f'y: {shape_str} event-wise static decision variables'
    assert y.event_adapt == [[0, 1, 4, 7, 8, 9], [3], [2, 5, 6]]
    new_linear = np.zeros((y.size, model.vt_model.last))
    new_linear[:y.size, y.first:y.last] = np.eye(y.size)
    assert (y.to_affine().linear.toarray() == new_linear).all()

    x.adapt(z)
    assert x.__repr__() == f'x: {length} affinely adaptive decision variables'
    assert x.event_adapt == [list(range(10))]
    new_linear = np.zeros((x.size, model.vt_model.last))
    new_linear[:x.size, x.first:x.last] = np.eye(x.size)
    assert (x.to_affine().linear.toarray() == new_linear).all()

    x.adapt(2)
    x.adapt([4, 5, 8])
    x.adapt(6)
    repr = f'x: {length} event-wise affinely adaptive decision variables'
    assert x.__repr__() == repr
    assert x.event_adapt == [[0, 1, 3, 7, 9], [2], [4, 5, 8], [6]]
    new_linear = np.zeros((x.size, model.vt_model.last))
    new_linear[:x.size, x.first:x.last] = np.eye(x.size)
    assert (x.to_affine().linear.toarray() == new_linear).all()


@pytest.mark.parametrize('shape1, shape2', [
    ((), (2,)),
    ((3,), ()),
    ((3,), (4,)),
    ((3, 5), (2, 3, 4))
])
def test_dvar_array_adapt(shape1, shape2):

    scens = ['A', 'B', 'C', 'D', 'E']
    model = dro.Model(scens)

    x = model.dvar(vtype='i')
    y = model.dvar(shape1)
    z = model.rvar(shape2)

    fset = model.ambiguity()

    x.adapt(['B', 'D'])
    s = fset.s
    y.adapt(s.loc[['A', 'C']])
    y.adapt(s['E'])
    y.adapt(z)

    if y.shape != () and z.shape != ():
        with pytest.raises(RuntimeError):
            y[0].adapt(z[0])

    num_dvar = 1 + 1 + int(np.prod(shape1))
    num_rvar = int(np.prod(shape2))
    var_exprs = model.rule_var()
    for expr in var_exprs:
        assert expr.__repr__() == f'{num_dvar} bi-affine expressions'
        num_all_var = 2 + 1*2
        num_all_var += int(np.prod(shape1))*3 + int(np.prod(shape1)*num_rvar)*3
        assert expr.raffine.linear.shape == (num_dvar*num_rvar, num_all_var)
        assert (expr.raffine.const == np.zeros((expr.size, num_rvar))).all()
        assert expr.affine.linear.shape[0] == num_dvar
        assert (expr.affine.const == np.zeros(num_dvar)).all()


def test_dvar_slice():

    model = dro.Model(10)
    x = model.dvar(5)
    xx = model.dvar((3, 4, 5))
    z = model.rvar(5)
    zz = model.rvar((4, 5))

    assert x[0].__repr__() == 'a static decision variable'
    assert x[:3].__repr__() == '3 static decision variables'
    assert xx[0].__repr__() == '4x5 static decision variables'
    assert xx[0, :, 2:4].__repr__() == '4x2 static decision variables'
    assert (xx[0] + 3).__repr__() == '4x5 affine expressions'
    repr = 'worst-case expectation of 4x5 affine expressions'
    assert (E(xx[0] + 3)).__repr__() == repr
    assert (xx[0] + 3 <= 0).__repr__() == '20 linear constraints'
    repr = '20 linear constraints of expectations'
    assert (E(xx[0] + 3) <= 0).__repr__() == repr
    assert (zz@x + 2).__repr__() == '4 bi-affine expressions'

    x.adapt(3)
    xx.adapt(z)
    assert x[0].__repr__() == 'an event-wise static decision variable'
    assert x[:3].__repr__() == '3 event-wise static decision variables'
    assert (x[:3] + 3).__repr__() == '3 event-wise affine expressions'
    repr = 'worst-case expectation of 3 affine expressions'
    assert (E(x[:3])).__repr__() == repr
    assert xx[0].__repr__() == '4x5 affinely adaptive decision variables'
    assert xx[0, :, 2:4].__repr__() == '4x2 affinely adaptive decision variables'
    assert (xx[0, :, 2:4] + 3).__repr__() == '4x2 bi-affine expressions'
    repr = 'worst-case expectation of 4x2 bi-affine expressions'
    assert (E(xx[0, :, 2:4] + 3)).__repr__() == repr
    assert (xx[0, :, 2:4] + 3 <= 0).__repr__() == '8 robust constraints'
    repr = '8 robust constraints of expectations'
    assert (E(xx[0, :, 2:4]) <= 0).__repr__() == repr
    assert (E(xx[0, :, 2:4]) >= 0).__repr__() == repr

    with pytest.raises(TypeError):
        x[:3].adapt(xx)

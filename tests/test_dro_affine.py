from rsome import dro
from rsome import E
from rsome import ort_solver as ort
import rsome as rso
import numpy as np
import numpy.random as rd
import pytest


def test_dvar_to_affine():

    model = dro.Model()
    x = model.dvar(5)
    xx = model.dvar((3, 4, 5))
    # z = model.rvar(5)
    zz = model.rvar((4, 5))

    affine1 = (x + 2)
    assert affine1.shape == (5, )
    assert affine1.__repr__() == '5 affine expressions'
    assert affine1[0].shape == ()
    assert affine1[0].__repr__() == 'an affine expression'

    affine2 = 2 * (xx*zz + 3)
    assert affine2.shape == (3, 4, 5)
    assert affine2.__repr__() == '3x4x5 bi-affine expressions'
    assert (affine2 == 0).__repr__() == '60 robust constraints'
    assert (E(affine2) <= 0).__repr__() == '60 robust constraints of expectations'


@pytest.mark.parametrize('array1, array2, const', [
    (rd.rand(1).reshape(()), rd.rand(1).reshape(()), 2.5),
    (rd.rand(7), rd.rand(7), rd.rand(7)),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7)),
    (rd.rand(2, 3, 7), rd.rand(7), rd.rand(3, 7)),
    (rd.rand(3, 4), rd.rand(3, 4), rd.rand(3, 4)),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3, 4)),
    (rd.rand(5), rd.rand(3, 5), -2.0),
    (rd.rand(2, 4, 2, 5), rd.rand(1, 2, 5), -np.arange(5)),
    (rd.rand(3, 4), rd.rand(2, 3, 4), rd.rand(3, 4))
])
def test_array_mul(array1, array2, const):

    shape = array2.shape
    target = array1*array2 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(shape)

    expr = array1*v + const
    m.min(a)
    m.st(a >= abs(expr - target), v == array2)
    m.solve(ort)

    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array2) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecAffine)
    if target.shape == ():
        shape_str = 'an'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} linear constraint{suffix}'


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(7), 3.5),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7), rd.rand(3, 1)),
    (rd.rand(3, 7, 4), rd.rand(4), rd.rand(4), rd.rand(7, 4)),
    (rd.rand(3, 7, 4), rd.rand(1, 7, 4), rd.rand(3, 7, 1), rd.rand(7, 4))
])
def test_random_array_add(array1, array2, array3, const):
    """
    This function tests summations of decision variables and random
    variables.
    """

    target = array1 + array2*array3 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array3.shape)

    expr = v + array2*z + const
    fset = m.ambiguity()
    fset.suppset(z == array3)
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(fset))
    m.st((target - expr <= d).forall(fset))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'

    with pytest.raises(ValueError):
        d.get(z)


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(6, 7), 3.5),
    (rd.rand(7, 2), rd.rand(7, 2), rd.rand(6, 7, 2), 3.5),
    (rd.rand(3, 7), rd.rand(7), rd.rand(5, 7), rd.rand(3, 1)),
    (rd.rand(3, 7, 4), rd.rand(4), rd.rand(3, 4), rd.rand(7, 4)),
    (rd.rand(2, 7, 3), rd.rand(1, 7, 3), rd.rand(3, 2, 7, 1), rd.rand(7, 3))
])
def test_event_exp_random_array_add(array1, array2, array3, const):
    """
    This function tests summations of decision variables and event-wise
    random variables.
    """

    ns = array3.shape[0]
    print(ns)

    target = array1 + (array2*(array3.sum(axis=0)/ns)) + const

    m = dro.Model(ns)
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array3.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array3[s])
    pr = m.p
    fset.probset(pr == 1/ns)

    expr = v + array2*z + const
    m.min(a)
    m.st(a >= d)
    m.st((E(expr) - target <= d).forall(fset))
    m.st((target - E(expr) <= d).forall(fset))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


@pytest.mark.parametrize('array1, array2, const', [
    (rd.rand(7), rd.rand(6, 7), 3.5),
    (rd.rand(7, 2), rd.rand(6, 7, 2), 3.5),
    (rd.rand(3, 7), rd.rand(5, 1, 7), rd.rand(3, 7)),
    (rd.rand(2, 5, 4), rd.rand(3, 2, 1, 4), rd.rand(5, 4)),
    (rd.rand(2, 7, 3), rd.rand(3, 2, 7, 1), rd.rand(7, 3))
])
def test_event_ro_random_array_add(array1, array2, const):

    ns = array2.shape[0]

    target = array1*array2 + const

    m = dro.Model(ns)
    a = m.dvar()
    d = m.dvar(target.shape[1:])

    z = m.rvar(array2.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array2[s])

    for s in range(ns):
        d.adapt(s)
    pr = m.p
    fset.probset(pr == 1/ns)

    expr = array1*z + const
    m.minsup(a, fset)
    m.st(a >= 0)
    m.st(d >= expr)
    m.st(d <= expr)
    m.solve(ort)

    for s in range(ns):
        assert (abs(d.get()[s] - target[s]) < 1e-4).all()

    suffix = 's' if target[0].size > 1 else ''
    repr = f'{target[0].size} event-wise robust constraint{suffix}'
    assert (expr <= d).__repr__() == repr


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(7), 3.5),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7), rd.rand(3, 1)),
    (rd.rand(3, 7, 4), rd.rand(4), rd.rand(4), rd.rand(7, 4)),
    (rd.rand(2, 7, 4), rd.rand(1, 7, 4), rd.rand(2, 7, 1), rd.rand(7, 4))
])
def test_random_array_mul(array1, array2, array3, const):
    """
    This function tests bi-affine expressions of decision variables
    and random variables.
    """

    target = array1*array2 + array3*array2 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array2.shape)

    expr = v*z + array3*z + const
    fset = m.ambiguity()
    fset.suppset(z == array2)
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(fset))
    m.st((target - expr <= d).forall(fset))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(6, 7), 3.5),
    (rd.rand(7, 2), rd.rand(7, 2), rd.rand(6, 7, 2), 3.5),
    (rd.rand(3, 7), rd.rand(7), rd.rand(5, 7), rd.rand(3, 1)),
    (rd.rand(3, 7, 4), rd.rand(4), rd.rand(2, 4), rd.rand(7, 4)),
    (rd.rand(2, 7, 3), rd.rand(1, 7, 3), rd.rand(5, 2, 7, 1), rd.rand(7, 3))
])
def test_event_exp_random_array_mul(array1, array2, array3, const):
    """
    This function tests multiplication of decision variables and event-wise
    random variables.
    """

    ns = array3.shape[0]

    target = array1*array2*(array3.sum(axis=0)/ns) + const

    m = dro.Model(ns)
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array3.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array3[s])
    pr = m.p
    fset.probset(pr == 1/ns)

    expr = v*array2*z + const
    m.min(a)
    m.st(a >= d)
    m.st((E(expr) - target <= d).forall(fset))
    m.st((target - E(expr) <= d).forall(fset))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


@pytest.mark.parametrize('array, const', [
    (rd.rand(), rd.rand()),
    (rd.rand(7), rd.rand()),
    (rd.rand(7), rd.rand(1)),
    (rd.rand(3, 7), rd.rand(7)),
    (rd.rand(3, 7), rd.rand(3, 7))
])
def test_random_adaptive_array_mul(array, const):

    m = dro.Model()
    a = m.dvar()
    shape = array.shape if not isinstance(array, float) else ()
    y = m.dvar(shape)

    cshape = const.shape if not isinstance(const, float) else ()
    z = m.rvar(cshape)
    uset = (z >= -100, z <= 100)

    y.adapt(z)

    m.min(a)
    m.st(a >= 0)
    m.st((y == array*z + const).forall(uset))
    m.solve(ort)

    y0 = y.get() * np.ones(cshape)
    yz = y.get(z).sum(axis=tuple(range(len(shape), len(shape+cshape))))

    assert (abs(y0 - const*np.ones(y0.shape)) <= 1e-4).all()
    assert (abs(yz - array) <= 1e-4).all()


@pytest.mark.parametrize('array1, array2, const', [
    (rd.rand(7), rd.rand(5, 7), 2.5),
    (rd.rand(7), rd.rand(5, 7), rd.rand(7)),
    (rd.rand(2, 3), rd.rand(5, 2, 1), rd.rand(1, 3)),
    (rd.rand(2, 3, 4), rd.rand(2, 2, 3, 4), rd.rand(3, 4))
])
def test_event_random_adaptive_array_mul(array1, array2, const):

    target = array1*array2 + const
    shape = target.shape[1:]
    zshape = array1.shape
    ns = target.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    v = m.dvar(array2.shape[1:])
    d = m.dvar(shape)

    z = m.rvar(zshape)
    u = m.rvar(array2.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z >= -100, z <= 100, u == array2[s])

    d.adapt(z)
    for s in range(ns):
        d.adapt(s)
        v.adapt(s)
    pr = m.p
    fset.probset(pr == 1/ns)

    expr = z*v + const
    m.minsup(a, fset)
    m.st(a >= 0)
    m.st(d >= expr)
    m.st(d <= expr)
    m.st(v == u)
    m.solve(ort)

    ones = np.zeros(array2.shape[1:])
    for s in range(ns):
        assert (abs(d.get()[s]*ones - const*ones) < 1e-4).all()
        assert (abs(d.get(z)[s].sum(axis=tuple(range(len(shape),
                                    len(shape+zshape))))*ones) < 1e-4).all()

    if target[0].shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target[0].shape])
        suffix = 's' if target[0].size > 1 else ''
    repr = f'{shape_str} event-wise bi-affine expression{suffix}'
    assert expr.__repr__() == repr
    repr = f'{target[0].size} event-wise robust constraint{suffix}'
    assert (expr <= 0).__repr__() == repr


@pytest.mark.parametrize('array1, array2, const', [
    (rd.rand(3), rd.rand(3), 1.5),
    (rd.rand(7), rd.rand(7, 3), rd.rand(3)),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(3, 4, 2)),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3)),
    (rd.rand(7, 3), rd.rand(3), rd.rand(7)),
    (rd.rand(4, 5), rd.rand(5, 3), -2.0),
    (rd.rand(4, 2, 5), rd.rand(5, 6), np.arange(6)),
    (rd.rand(2, 4, 2, 5), rd.rand(4, 5, 3), -np.arange(3)),
    (rd.rand(4, 5), rd.rand(2, 5, 3), np.arange(12).reshape(4, 3))
])
def test_mat_rmul(array1, array2, const):
    """
    This function tests a numeric array matmul a variable array
    """

    target = array1@array2 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)

    expr = array1@v + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array2)
    m.solve(ort)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array2) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecAffine)


@pytest.mark.parametrize('array1, array2, const', [
    (rd.rand(3), rd.rand(3), 1.5),
    (rd.rand(7), rd.rand(7, 3), rd.rand(3)),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(3, 4, 2)),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3)),
    (rd.rand(7, 3), rd.rand(3), rd.rand(7)),
    (rd.rand(4, 5), rd.rand(5, 3), -2.0),
    (rd.rand(4, 2, 5), rd.rand(5, 6), np.arange(6)),
    (rd.rand(2, 4, 2, 5), rd.rand(4, 5, 3), -np.arange(3)),
    (rd.rand(4, 5), rd.rand(2, 5, 3), np.arange(12).reshape(4, 3))
])
def test_mat_mul(array1, array2, const):
    """
    This function tests a variable array matmul a numeric array
    """

    target = array1@array2 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    expr = v@array2 + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecAffine)


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(7), 3.5),
    (rd.rand(7), rd.rand(7, 6), rd.rand(7), 2.5),
    (rd.rand(7), rd.rand(7, 6), rd.rand(7, 6), rd.rand(2, 6)),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(4, 7, 2), rd.rand(4, 2)),
    (rd.rand(3, 4), rd.rand(4), rd.rand(4), rd.rand(3)),
    (rd.rand(3, 4, 5, 6), rd.rand(6), rd.rand(6), rd.rand(5)),
    (rd.rand(3, 5), rd.rand(5, 2), rd.rand(5, 2), rd.rand(3, 2)),
    (rd.rand(3, 4, 5, 6), rd.rand(6, 3), rd.rand(6, 3), rd.rand(3)),
    (rd.rand(3, 4, 5, 6), rd.rand(4, 6, 2), rd.rand(6, 2), rd.rand(5, 2))
])
def test_random_mat_mul(array1, array2, array3, const):
    """
    This function tests bi-affine expressions created by random varaibles
    matmul decision variables.
    """

    target = array1@array2 + array1@array3 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array1.shape)

    expr = z@v + z@array3 + const
    uset = (abs(z - array1) <= 0)
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(uset))
    m.st((target - expr <= d).forall(uset))
    m.st(v == array2)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array2) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(7), 3.5),
    (rd.rand(7), rd.rand(7, 6), rd.rand(7), 2.5),
    (rd.rand(7), rd.rand(7, 6), rd.rand(7, 6), rd.rand(2, 6)),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(4, 7, 2), rd.rand(4, 2)),
    (rd.rand(3, 4), rd.rand(4), rd.rand(4), rd.rand(3)),
    (rd.rand(3, 4, 5, 6), rd.rand(6), rd.rand(6), rd.rand(5)),
    (rd.rand(3, 5), rd.rand(5, 2), rd.rand(5, 2), rd.rand(3, 2)),
    (rd.rand(3, 4, 5, 6), rd.rand(6, 3), rd.rand(6, 3), rd.rand(3)),
    (rd.rand(3, 4, 5, 6), rd.rand(4, 6, 2), rd.rand(6, 2), rd.rand(5, 2))
])
def test_mat_random_mul(array1, array2, array3, const):
    """
    This function tests bi-affine expressions created by decision
    varaibles matmul decision variables.
    """

    target = array1@array2 + array1@array3 + const

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array2.shape)

    expr = v@z + v@array3 + const
    uset = (z == array2, )
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(uset))
    m.st((target - expr <= d).forall(uset))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)


@pytest.mark.parametrize('array1, array2, const, array3', [
    (rd.rand(7), rd.rand(7), rd.rand(7), rd.rand(7)),
    (rd.rand(7, 3), rd.rand(3), rd.rand(), rd.rand(3, 4)),
    (rd.rand(7), rd.rand(5, 7), rd.rand(5, 1), rd.rand(7, 2)),
    (rd.rand(2, 3, 4), rd.rand(3, 4), rd.rand(4), rd.rand(4)),
    (rd.rand(2, 3, 4), rd.rand(3, 4), rd.rand(4), rd.rand(2, 4, 3)),
])
def test_roaffine_mat_mul(array1, array2, const, array3):

    target = (array1*array2 + const)@array3

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)
    z = m.rvar(array1.shape)

    expr = (z*v + const)@array3
    fset = m.ambiguity()
    fset.suppset(z == array1)
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(fset))
    m.st((target - expr <= d).forall(fset))
    m.st(v == array2)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array2) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)


@pytest.mark.parametrize('array1, array2, const, array3', [
    (rd.rand(7), rd.rand(7), rd.rand(7), rd.rand(7)),
    (rd.rand(7, 3), rd.rand(3), rd.rand(), rd.rand(3, 7)),
    (rd.rand(7), rd.rand(5, 7), rd.rand(5, 1), rd.rand(3, 5)),
    (rd.rand(2, 3, 4), rd.rand(3, 4), rd.rand(4), rd.rand(3)),
    (rd.rand(2, 3, 4), rd.rand(3, 4), rd.rand(4), rd.rand(2, 3, 3))
])
def test_mat_roaffine_mul(array1, array2, const, array3):

    target = array3@(array1*array2 + const)

    m = dro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)
    z = m.rvar(array1.shape)

    expr = array3@(z*v + const)
    fset = m.ambiguity()
    fset.suppset(z == array1)
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(fset))
    m.st((target - expr <= d).forall(fset))
    m.st(v == array2)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array2) < 1e-4).all()
    assert isinstance(expr, rso.lp.DecRoAffine)


def test_affine_errors():

    model = dro.Model()
    x = model.dvar((1, 6))
    y = model.dvar((6, 6))
    z = model.rvar(6)

    y.adapt(z)

    with pytest.raises(TypeError):
        x * x

    with pytest.raises(TypeError):
        x @ x.T

    with pytest.raises(TypeError):
        x @ y

    with pytest.raises(TypeError):
        y @ x.T

    with pytest.raises(TypeError):
        x @ np.array(['C']*6)

    with pytest.raises(TypeError):
        y * z

    with pytest.raises(TypeError):
        z * y

    with pytest.raises(TypeError):
        y @ z

    with pytest.raises(TypeError):
        (y + 2) @ (z + 1)

    with pytest.raises(TypeError):
        z @ y

from rsome import ro
from rsome import ort_solver as ort
import numpy as np
import numpy.random as rd
import pytest


def test_scalar_opt():

    model = ro.Model()

    x = model.dvar(vtype='i')
    y = model.ldr()
    z = model.rvar()

    affine1 = 2*x + y + 2.0
    assert type(affine1) == ro.Affine
    assert affine1.const == 2
    assert affine1.__repr__() == 'an affine expression'
    assert (affine1 <= 0).__repr__() == '1 linear constraint'

    neg_affine1 = - affine1
    assert type(neg_affine1) == ro.Affine
    assert (neg_affine1.linear.toarray() == - affine1.linear.toarray()).all()
    assert neg_affine1.const == - affine1.const
    assert neg_affine1.__repr__() == 'an affine expression'

    array1 = 3.5*affine1 - 0.5
    assert type(array1) == ro.Affine
    assert (array1.linear.toarray() == 3.5*affine1.linear.toarray()).all()
    assert array1.const == 3.5*affine1.const - 0.5
    assert array1.__repr__() == 'an affine expression'
    assert (array1 <= 0).__repr__() == '1 linear constraint'

    array2 = np.arange(1, 5)*affine1 - 0.5
    assert type(array2) == ro.Affine
    new_linear = np.arange(1, 5).reshape((4, 1))*affine1.linear.toarray()
    assert (array2.linear.toarray() == new_linear).all()
    assert (array2.const == np.arange(1, 5)*affine1.const - 0.5).all()
    assert array2.__repr__() == '4 affine expressions'
    assert (array2 <= 0).__repr__() == '4 linear constraints'

    array3 = affine1 - np.arange(1, 5)
    assert type(array3) == ro.Affine
    new_linear = np.ones((4, 1))*affine1.linear.toarray()
    assert (array3.linear.toarray() == affine1.linear.toarray()).all()
    assert (array3.const == affine1.const - np.arange(1, 5)).all()
    assert array3.__repr__() == '4 affine expressions'
    assert (array3 <= 0).__repr__() == '4 linear constraints'

    array4 = affine1 - 3*z
    assert type(array4) == ro.RoAffine
    assert (array4.raffine.linear.toarray() == 0).all()
    assert (array4.raffine.const == -3).all()
    assert (array4.affine.linear.toarray() == affine1.linear.toarray()).all()
    assert (array4.affine.const == affine1.const).all()
    assert array4.__repr__() == 'a bi-affine expression'
    assert (array4 <= 0).__repr__() == '1 robust constraint'

    array5 = affine1*z - z + 3.2
    assert type(array5) == ro.RoAffine
    assert (array5.raffine.linear.toarray() == affine1.linear.toarray()).all()
    assert (array5.raffine.const == affine1.const - 1).all()
    assert (array5.affine.linear.toarray() == 0).all()
    assert (array5.affine.const == 3.2).all()
    # assert (array5.raffine.linear =)
    assert array5.__repr__() == 'a bi-affine expression'
    assert (array5 <= 0).__repr__() == '1 robust constraint'

    array6 = array5 * rd.rand(5) + 3
    assert array6.__repr__() == '5 bi-affine expressions'
    assert (array6 == 1).__repr__() == '5 robust constraints'

    array7 = array6 @ rd.rand(5) + 3
    assert array7.__repr__() == 'a bi-affine expression'
    assert (array7 == 1).__repr__() == '1 robust constraint'


@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(1).reshape(()), rd.rand(1).reshape(()), 2.5, ort),
    (rd.rand(7), rd.rand(7), rd.rand(7), ort),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7), ort),
    (rd.rand(2, 3, 7), rd.rand(7), rd.rand(3, 7), ort),
    (rd.rand(3, 4), rd.rand(3, 4), rd.rand(3, 4), ort),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3, 4), ort),
    (rd.rand(5), rd.rand(3, 5), -2.0, ort),
    (rd.rand(2, 4, 2, 5), rd.rand(1, 2, 5), -np.arange(5), ort),
    (rd.rand(3, 4), rd.rand(2, 3, 4), rd.rand(3, 4), ort)
])
def test_array_mul(array1, array2, const, solver):
    """
    This function tests a variable array times a numeric array
    """

    target = array1*array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)

    expr = array1*v + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array2)
    m.solve(solver)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array2) < 1e-4).all()
    assert type(expr) == ro.Affine
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

    m = ro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array3.shape)

    expr = v + array2*z + const
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(z + (-array3) == 0))
    m.st((target - expr <= d).forall((-array3) + z == 0))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert type(expr) == ro.RoAffine
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


@pytest.mark.parametrize('array1, array2, array3, const', [
    (rd.rand(7), rd.rand(7), rd.rand(7), 3.5),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7), rd.rand(3, 1)),
    (rd.rand(3, 7, 4), rd.rand(4), rd.rand(4), rd.rand(7, 4)),
    (rd.rand(3, 7, 4), rd.rand(1, 7, 4), rd.rand(3, 7, 1), rd.rand(7, 4))
])
def test_random_array_mul(array1, array2, array3, const):
    """
    This function tests bi-affine expressions of decision variables
    and random variables.
    """

    target = array1*array2 + array3*array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array2.shape)

    expr = v*z + array3*z + const
    uset = (z == array2, )
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(uset))
    m.st((target - expr <= d).forall(uset))
    m.st(v == array1)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array1) < 1e-4).all()
    assert type(expr) == ro.RoAffine
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(3), rd.rand(3), 1.5, ort),
    (rd.rand(7), rd.rand(7, 3), rd.rand(3), ort),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(3, 4, 2), ort),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3), ort),
    (rd.rand(7, 3), rd.rand(3), rd.rand(7), ort),
    (rd.rand(4, 5), rd.rand(5, 3), -2.0, ort),
    (rd.rand(4, 2, 5), rd.rand(5, 6), np.arange(6), ort),
    (rd.rand(2, 4, 2, 5), rd.rand(4, 5, 3), -np.arange(3), ort),
    (rd.rand(4, 5), rd.rand(2, 5, 3), np.arange(12).reshape(4, 3), ort)
])
def test_mat_rmul(array1, array2, const, solver):
    """
    This function tests a numeric array matmul a variable array
    """

    target = array1@array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)

    expr = array1@v + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array2)
    m.solve(solver)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array2) < 1e-4).all()
    assert type(expr) == ro.Affine
    if target.shape == ():
        shape_str = 'an'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} linear constraint{suffix}'


@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(3), rd.rand(3), 1.5, ort),
    (rd.rand(7), rd.rand(7, 3), rd.rand(3), ort),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(3, 4, 2), ort),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3), ort),
    (rd.rand(7, 3), rd.rand(3), rd.rand(7), ort),
    (rd.rand(4, 5), rd.rand(5, 3), -2.0, ort),
    (rd.rand(4, 2, 5), rd.rand(5, 6), np.arange(6), ort),
    (rd.rand(2, 4, 2, 5), rd.rand(4, 5, 3), -np.arange(3), ort),
    (rd.rand(4, 5), rd.rand(2, 5, 3), np.arange(12).reshape(4, 3), ort)
])
def test_mat_mul(array1, array2, const, solver):
    """
    This function tests a variable array matmul a numeric array
    """

    target = array1@array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.dvar(array1.shape)
    d = m.dvar(target.shape)

    expr = v@array2 + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array1)
    m.solve(solver)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array1) < 1e-4).all()
    assert type(expr) == ro.Affine
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

    m = ro.Model()
    a = m.dvar()
    v = m.dvar(array2.shape)
    d = m.dvar(target.shape)

    z = m.rvar(array1.shape)

    expr = z@v + z@array3 + const
    uset = (z == array1, )
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(uset))
    m.st((target - expr <= d).forall(uset))
    m.st(v == array2)
    m.solve(ort)
    assert abs(m.get()) < 1e-3
    assert (abs(v.get() - array2) < 1e-4).all()
    assert type(expr) == ro.RoAffine
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


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

    m = ro.Model()
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
    assert type(expr) == ro.RoAffine
    if target.shape == ():
        shape_str = 'a'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} bi-affine expression{suffix}'
    assert (expr <= 0).__repr__() == f'{target.size} robust constraint{suffix}'


def test_affine_errors():

    model = ro.Model()
    x = model.dvar((1, 6))
    y = model.dvar((6, 6))
    r = model.ldr((6, 6))

    with pytest.raises(TypeError):
        x * y

    with pytest.raises(TypeError):
        x * r

    with pytest.raises(TypeError):
        x @ y

    with pytest.raises(TypeError):
        x @ r

    with pytest.raises(TypeError):
        x @ np.array(['C']*6)

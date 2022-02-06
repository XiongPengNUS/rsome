from rsome import ro
from rsome import lpg_solver as lpg
from rsome import ort_solver as ort
from rsome import clp_solver as clp
from rsome import cvx_solver as cvx
from rsome import grb_solver as grb
import numpy as np
import numpy.random as rd
import scipy.sparse as sp
from scipy.linalg import block_diag
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

    array2 = np.arange(1, 5)*affine1 - 0.5
    assert type(array2) == ro.Affine
    new_linear = np.arange(1, 5).reshape((4, 1))*affine1.linear.toarray()
    assert (array2.linear.toarray() == new_linear).all()
    assert (array2.const == np.arange(1, 5)*affine1.const - 0.5).all()
    assert array2.__repr__() == '4 affine expressions'

    array3 = affine1 - np.arange(1, 5)
    assert type(array3) == ro.Affine
    new_linear = np.ones((4, 1))*affine1.linear.toarray()
    assert (array3.linear.toarray() == affine1.linear.toarray()).all()
    assert (array3.const == affine1.const - np.arange(1, 5)).all()
    assert array3.__repr__() == '4 affine expressions'

    array4 = affine1 - 3*z
    assert type(array4) == ro.RoAffine
    assert (array4.raffine.linear.toarray() == 0).all()
    assert (array4.raffine.const == -3).all()
    assert (array4.affine.linear.toarray() == affine1.linear.toarray()).all()
    assert (array4.affine.const == affine1.const).all()
    assert array4.__repr__() == 'a bi-affine expression'

    array5 = affine1*z - z + 3.2
    assert type(array5) == ro.RoAffine
    assert (array5.raffine.linear.toarray() == affine1.linear.toarray()).all()
    assert (array5.raffine.const == affine1.const - 1).all()
    assert (array5.affine.linear.toarray() == 0).all()
    assert (array5.affine.const == 3.2).all()
    # assert (array5.raffine.linear =)
    assert array5.__repr__() == 'a bi-affine expression'


'''
def test_scalar_ldr_opt():

    model = ro.Model()
    x = model.dvar()
    y = model.ldr()

    z = model.rvar(3)
    y.adapt(z[0])

    affine1 = 2*x + y + 2.0
    assert type(affine1) == ro.RoAffine
    assert affine1.__repr__() == 'a bi-affine expression'

    array1 = -2.5*affine1 - 0.5
    assert type(array1) == ro.RoAffine
    new_linear = -2.5 * affine1.raffine.linear.toarray()
    assert (array1.raffine.linear.toarray() == new_linear).all()
    assert (array1.raffine.const == -2.5 * affine1.raffine.const).all()
    new_linear = -2.5 * affine1.affine.linear.toarray()
    assert (array1.affine.linear.toarray() == new_linear).all()
    assert (array1.affine.const == -2.5*affine1.affine.const - 0.5)

    with pytest.raises(TypeError):
        z * y - x
'''


@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(1).reshape(()), rd.rand(1).reshape(()), 2.5, lpg),
    (rd.rand(7), rd.rand(7), rd.rand(7), ort),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7), ort),
    (rd.rand(2, 3, 7), rd.rand(7), rd.rand(3, 7), clp),
    (rd.rand(3, 4), rd.rand(3, 4), rd.rand(3, 4), clp),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3, 4), cvx),
    (rd.rand(5), rd.rand(3, 5), -2.0, cvx),
    (rd.rand(2, 4, 2, 5), rd.rand(1, 2, 5), -np.arange(5), grb),
    (rd.rand(3, 4), rd.rand(2, 3, 4), rd.rand(3, 4), grb)
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

'''
@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(1).reshape(()), rd.rand(1).reshape(()), 2.5, lpg),
    (rd.rand(7), rd.rand(7), rd.rand(7), ort),
    (rd.rand(3, 7), rd.rand(7), rd.rand(7), ort),
    (rd.rand(2, 3, 7), rd.rand(7), rd.rand(3, 7), clp),
    (rd.rand(3, 4), rd.rand(3, 4), rd.rand(3, 4), clp),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3, 4), cvx),
    (rd.rand(5), rd.rand(3, 5), -2.0, cvx),
    (rd.rand(2, 4, 2, 5), rd.rand(1, 2, 5), -np.arange(5), grb),
    (rd.rand(3, 4), rd.rand(2, 3, 4), rd.rand(3, 4), grb)
])
def test_ldr_mul(array1, array2, const, solver):
    """
    This function tests a variable array times a numeric array
    """

    target = array1*array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.ldr(array2.shape)
    d = m.dvar(target.shape)

    expr = array1*v + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array2)
    m.solve(solver)
    assert abs(m.get()) < 1e-4
    assert type(expr) == ro.Affine
    if target.shape == ():
        shape_str = 'an'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} affine expression{suffix}'
'''


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
    uset = (z == array3, )
    m.min(a)
    m.st(a >= d)
    m.st((expr - target <= d).forall(uset))
    m.st((target - expr <= d).forall(uset))
    m.st(v == array1)
    m.solve(grb)
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
    m.solve(grb)
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


'''
def test_array_ldr_opt():

    model = ro.Model()

    x = model.dvar(5)
    y = model.ldr(5)
    yy = model.ldr((5, 3))

    z = model.rvar(8)
    y.adapt(z)
    yy[:, 2].adapt(z[1])

    affine1 = x + 3*y - 2
    assert type(affine1) == ro.RoAffine
    new_linear = 3*y.to_affine().raffine.linear.toarray()
    assert (affine1.raffine.linear.toarray() == new_linear).all()
    assert (affine1.raffine.const == np.zeros((5, 8))).all()
    new_linear = (x  + 3*y.fixed).linear.toarray()
    assert (affine1.affine.linear.toarray() == new_linear).all()
    assert (affine1.affine.const == -2 * np.ones(5)).all()
    assert affine1.__repr__() == '5 bi-affine expressions'

    affine2 = x - 3*yy[:, 0] + 2.5
    assert type(affine1) == ro.RoAffine
    assert (affine2.raffine.linear.toarray() == 0).all()
    assert (affine2.raffine.const == np.zeros((5, 8))).all()
    new_linear = (x - 3*yy.fixed[:, 0]).linear.toarray()
    assert (affine2.affine.linear == new_linear).all()
    assert (affine2.affine.const == 2.5 * np.ones(5)).all()
    assert affine2.__repr__() == '5 bi-affine expressions'

    affine3 = x*z[:5] + y - z[:5] + 1.5
    assert type(affine3) == ro.RoAffine
    new_linear = (x*z[:5] + y).raffine.linear.toarray()
    assert (affine3.raffine.linear.toarray() == new_linear).all()
    new_const = - sp.csr_matrix((np.ones(5),
                                 np.arange(5),
                                 np.arange(6)), shape=(5, 8)).toarray()
    assert (affine3.raffine.const == new_const).all()
    new_linear = (y.fixed).to_affine().linear.toarray()
    assert (affine3.affine.linear.toarray() == new_linear).all()
    assert (affine3.affine.const == 1.5 * np.ones(5)).all()
    assert affine3.__repr__() == '5 bi-affine expressions'

    affine4 = x*z[:5] + yy.T -2.5*x + 3.5
    assert type(affine4) == ro.RoAffine
    new_linear = (x*z[:5] + yy.T).raffine.linear.toarray()
    assert (affine4.raffine.linear.toarray() == new_linear).all()
    assert (affine4.raffine.const == np.zeros((15, 8))).all()
    new_linear = (yy.T - 2.5*x).affine.linear.toarray()
    assert (affine4.affine.linear.toarray() == new_linear).all()
    assert (affine4.affine.const == 3.5 * np.ones((3, 5))).all()
    assert affine4.__repr__() == '3x5 bi-affine expressions'

    affine5 = x*z[:5] + yy[:, :1].T -2.5*x + 3.5
    assert type(affine5) == ro.RoAffine
    assert affine5.__repr__() == '1x5 bi-affine expressions'

    with pytest.raises(TypeError):
        z[-5:] * y - x

    with pytest.raises(TypeError):
        x * affine1

    with pytest.raises(TypeError):
        z[:5] * affine1

    coef_array = np.arange(1, 6)
    array1 = coef_array * affine1 + x
    assert type(array1) == ro.RoAffine
    new_linear = (np.diag(coef_array) @ affine1.raffine).linear.toarray()
    new_const = (np.diag(coef_array) @ affine1.raffine).const
    assert (array1.raffine.linear.toarray() == new_linear).all()
    assert (array1.raffine.const == new_const).all()
    new_linear = np.diag(coef_array) @ affine1.affine.linear.toarray()
    if new_linear.shape[1] < array1.affine.linear.shape[1]:
        extra_zeros = np.zeros((new_linear.shape[0],
                                array1.affine.linear.shape[1] - new_linear.shape[1]))
        new_linear = np.concatenate((new_linear, extra_zeros), axis=1)
    new_linear += x.to_affine().linear.toarray()
    assert (array1.affine.linear.toarray() == new_linear).all()
    assert (array1.affine.const == coef_array * affine1.affine.const).all()
    assert array1.__repr__() == '5 bi-affine expressions'

    coef_array = np.arange(1, 16).reshape((3, 5))
    array2 = coef_array * affine2 + x + 3.5
    assert type(array2) == ro.RoAffine
    temp = np.concatenate([np.diag(np.arange(i, i+5)) for i in range(0, 15, 5)])
    new_linear = (temp @ affine2.raffine).linear.toarray()
    new_const = (temp @ affine2.raffine).const
    assert (array2.raffine.linear.toarray() == new_linear).all()
    assert (array2.raffine.const == new_const).all()
    new_linear = ((coef_array * affine2.affine) + x).linear.toarray()
    new_const = coef_array * affine2.affine.const + 3.5
    assert (array2.affine.linear.toarray() == new_linear).all()
    assert (array2.affine.const == new_const).all()
    assert array2.__repr__() == '3x5 bi-affine expressions'

    coef_array = np.arange(1, 6)
    index = 1
    array3 = coef_array * affine4[index] + x - 0.5
    slice = np.arange(index*5, (index+1)*5, dtype=int)
    assert type(array3) == ro.RoAffine
    new_linear = (np.diag(coef_array) @ affine4.raffine[slice]).linear.toarray()
    new_const = (np.diag(coef_array) @ affine4.raffine[slice]).const
    assert (array3.raffine.linear.toarray() == new_linear).sum()
    assert (array3.raffine.const == new_const).sum()
    new_linear = (coef_array * affine4.affine[index] + x).linear.toarray()
    assert (array3.affine.linear.toarray() == new_linear).all()
    assert (array3.affine.const == coef_array*affine4.affine.const[index] - 0.5).all()
    assert array3.__repr__() == '5 bi-affine expressions'
'''


@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(3), rd.rand(3), 1.5, lpg),
    (rd.rand(7), rd.rand(7, 3), rd.rand(3), ort),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(3, 4, 2), ort),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3), clp),
    (rd.rand(7, 3), rd.rand(3), rd.rand(7), clp),
    (rd.rand(4, 5), rd.rand(5, 3), -2.0, cvx),
    (rd.rand(4, 2, 5), rd.rand(5, 6), np.arange(6), cvx),
    (rd.rand(2, 4, 2, 5), rd.rand(4, 5, 3), -np.arange(3), grb),
    (rd.rand(4, 5), rd.rand(2, 5, 3), np.arange(12).reshape(4, 3), grb)
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


@pytest.mark.parametrize('array1, array2, const, solver', [
    (rd.rand(3), rd.rand(3), 1.5, lpg),
    (rd.rand(7), rd.rand(7, 3), rd.rand(3), ort),
    (rd.rand(7), rd.rand(3, 4, 7, 2), rd.rand(3, 4, 2), ort),
    (rd.rand(2, 3, 4), rd.rand(4), rd.rand(3), clp),
    (rd.rand(7, 3), rd.rand(3), rd.rand(7), clp),
    (rd.rand(4, 5), rd.rand(5, 3), -2.0, cvx),
    (rd.rand(4, 2, 5), rd.rand(5, 6), np.arange(6), cvx),
    (rd.rand(2, 4, 2, 5), rd.rand(4, 5, 3), -np.arange(3), grb),
    (rd.rand(4, 5), rd.rand(2, 5, 3), np.arange(12).reshape(4, 3), grb)
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
    m.solve(ort)
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
    m.solve(grb)
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
    m.solve(grb)
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

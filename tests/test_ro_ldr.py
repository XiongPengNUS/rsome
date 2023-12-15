from rsome import ro
from rsome import grb_solver as grb
import numpy as np
import numpy.random as rd
import scipy.sparse as sp
import pytest


def test_ldr_scalar():

    model = ro.Model()
    z = model.rvar()

    x = model.ldr(name='x')
    assert x.shape == ()
    assert x.size == 1
    assert x.__repr__() == 'x: a decision rule variable'

    y = model.ldr(name='y')
    y.adapt(z)
    assert y.shape == ()
    assert y.size == 1
    assert y.__repr__() == 'y: a decision rule variable'

    xa = x.to_affine()
    assert xa.shape == ()
    assert xa.size == 1
    assert xa.const == 0
    assert xa.__repr__() == 'an affine expression'

    ya = y.to_affine()
    assert ya.shape == ()
    assert ya.size == 1
    assert ya.__repr__() == 'a bi-affine expression'

    with pytest.raises(TypeError):
        model.ldr(vtype='I')

    with pytest.raises(TypeError):
        x * y

    with pytest.raises(TypeError):
        z * y


def test_ldr_adapt():

    model = ro.Model()
    z = model.rvar((4, 7))
    x = model.dvar(5)
    y = model.ldr((3, 5))
    assert y.__repr__() == '3x5 decision rule variables'
    ya = y.to_affine()
    assert ya.shape == (3, 5)
    assert ya.size == 15
    i = range(6, 21)
    new_linear = np.zeros((15, model.rc_model.last))
    new_linear[range(15), i] = 1
    assert (ya.linear.toarray() == new_linear).all()
    assert (ya.const == 0).all()
    assert ya.__repr__() == '3x5 affine expressions'
    assert (x == 0).__repr__() == '5 linear constraints'

    model = ro.Model()
    z = model.rvar((4, 7))
    x = model.dvar(5)
    y = model.ldr((3, 5))
    y[1].adapt(z[1])
    ya = y.to_affine()
    assert ya.shape == (3, 5)
    assert ya.size == 15
    nr = model.sup_model.last
    new_linear = np.zeros((15*nr, model.rc_model.last))
    i = range(21, 56)
    j = (np.arange(5, 10, dtype=int).reshape((5, 1))*nr +
         np.arange(7, 14, dtype=int)).flatten()
    new_linear[j, i] = 1
    assert (ya.raffine.linear.toarray() == new_linear).all()
    assert (ya.raffine.const == 0).all()
    i = range(6, 21)
    new_linear = np.zeros((15, model.rc_model.last))
    new_linear[range(15), i] = 1
    assert (ya.affine.linear.toarray() == new_linear).all()
    assert (ya.affine.const == 0).all()
    assert ya.__repr__() == '3x5 bi-affine expressions'
    assert (x == 0).__repr__() == '5 linear constraints'

    model = ro.Model()
    z = model.rvar((4, 7))
    x = model.dvar(5)
    y = model.ldr((3, 5))
    y[1].adapt(z[1:3])
    ya = y.to_affine()
    assert ya.shape == (3, 5)
    assert ya.size == 15
    nr = model.sup_model.last
    new_linear = np.zeros((15*nr, model.rc_model.last))
    i = range(21, 91)
    j = (np.arange(5, 10, dtype=int).reshape((5, 1))*nr +
         np.arange(7, 21, dtype=int)).flatten()
    new_linear[j, i] = 1
    assert (ya.raffine.linear.toarray() == new_linear).all()
    assert (ya.raffine.const == 0).all()
    i = range(6, 21)
    new_linear = np.zeros((15, model.rc_model.last))
    new_linear[range(15), i] = 1
    assert (ya.affine.linear.toarray() == new_linear).all()
    assert (ya.affine.const == 0).all()
    assert ya.__repr__() == '3x5 bi-affine expressions'
    assert (x == 0).__repr__() == '5 linear constraints'

    model = ro.Model()
    z = model.rvar((4, 7))
    y = model.ldr((3, 5))
    y[1].adapt(z[1:3])
    with pytest.raises(RuntimeError):
        y[1:3].adapt(z[:, 0])


def test_scalar_ldr_opt():

    model = ro.Model()
    x = model.dvar()
    y = model.ldr()

    z = model.rvar(3)
    y.adapt(z[0])

    affine1 = 2*x + y + 2.0
    assert isinstance(affine1, ro.RoAffine)
    assert affine1.__repr__() == 'a bi-affine expression'

    array1 = -2.5*affine1 - 0.5
    assert isinstance(array1, ro.RoAffine)
    new_linear = -2.5 * affine1.raffine.linear.toarray()
    assert (array1.raffine.linear.toarray() == new_linear).all()
    assert (array1.raffine.const == -2.5 * affine1.raffine.const).all()
    new_linear = -2.5 * affine1.affine.linear.toarray()
    assert (array1.affine.linear.toarray() == new_linear).all()
    assert (array1.affine.const == -2.5*affine1.affine.const - 0.5)

    with pytest.raises(TypeError):
        z*y - x


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
def test_ldr_mul(array1, array2, const):
    """
    This function tests a fixed decision rule times a numeric array
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
    m.solve(grb)
    assert abs(m.get()) < 1e-4
    assert isinstance(expr, ro.Affine)
    if target.shape == ():
        shape_str = 'an'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} affine expression{suffix}'


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
def test_ldr_mat_rmul(array1, array2, const):
    """
    This function tests a numeric array matmul a fixed decision rule
    """

    target = array1@array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.ldr(array2.shape)
    d = m.dvar(target.shape)

    expr = array1@v + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array2)
    m.solve(grb)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array2) < 1e-4).all()
    assert isinstance(expr, ro.Affine)
    if target.shape == ():
        shape_str = 'an'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} affine expression{suffix}'


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
def test_ldr_mat_mul(array1, array2, const):
    """
    This function tests a fixed decision rule matmul a numeric array
    """

    target = array1@array2 + const

    m = ro.Model()
    a = m.dvar()
    v = m.ldr(array1.shape)
    d = m.dvar(target.shape)

    expr = v@array2 + const
    m.min(a)
    m.st(a >= abs(d))
    m.st(d == expr - target)
    m.st(v == array1)
    m.solve(grb)
    assert abs(m.get()) < 1e-4
    assert (abs(v.get() - array1) < 1e-4).all()
    assert isinstance(expr, ro.Affine)
    if target.shape == ():
        shape_str = 'an'
        suffix = ''
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
    assert expr.__repr__() == f'{shape_str} affine expression{suffix}'


def test_array_ldr_opt():

    model = ro.Model()

    x = model.dvar(5)
    y = model.ldr(5)
    yy = model.ldr((5, 3))

    z = model.rvar(8)
    y.adapt(z)
    yy[:, 2].adapt(z[1])

    affine1 = x + 3*y - 2
    assert isinstance(affine1, ro.RoAffine)
    new_linear = 3*y.to_affine().raffine.linear.toarray()
    assert (affine1.raffine.linear.toarray() == new_linear).all()
    assert (affine1.raffine.const == np.zeros((5, 8))).all()
    new_linear = (x + 3*y.fixed).linear.toarray()
    assert (affine1.affine.linear.toarray() == new_linear).all()
    assert (affine1.affine.const == -2 * np.ones(5)).all()
    assert affine1.__repr__() == '5 bi-affine expressions'

    affine2 = x - 3*yy[:, 0] + 2.5
    assert isinstance(affine1, ro.RoAffine)
    assert (affine2.raffine.linear.toarray() == 0).all()
    assert (affine2.raffine.const == np.zeros((5, 8))).all()
    new_linear = (x - 3*yy.fixed[:, 0]).linear.toarray()
    assert (affine2.affine.linear == new_linear).all()
    assert (affine2.affine.const == 2.5 * np.ones(5)).all()
    assert affine2.__repr__() == '5 bi-affine expressions'

    affine3 = x*z[:5] + y*(-2) - z[:5] + 1.5
    assert isinstance(affine3, ro.RoAffine)
    new_linear = (x*z[:5] - 2*y).raffine.linear.toarray()
    assert (affine3.raffine.linear.toarray() == new_linear).all()
    new_const = - sp.csr_matrix((np.ones(5),
                                 np.arange(5),
                                 np.arange(6)), shape=(5, 8)).toarray()
    assert (affine3.raffine.const == new_const).all()
    new_linear = (-2*y.fixed).to_affine().linear.toarray()
    assert (affine3.affine.linear.toarray() == new_linear).all()
    assert (affine3.affine.const == 1.5 * np.ones(5)).all()
    assert affine3.__repr__() == '5 bi-affine expressions'

    affine4 = x*z[:5] + yy.T - 2.5*x + 3.5
    assert isinstance(affine4, ro.RoAffine)
    new_linear = (x*z[:5] + yy.T).raffine.linear.toarray()
    assert (affine4.raffine.linear.toarray() == new_linear).all()
    assert (affine4.raffine.const == np.zeros((15, 8))).all()
    new_linear = (yy.T - 2.5*x).affine.linear.toarray()
    assert (affine4.affine.linear.toarray() == new_linear).all()
    assert (affine4.affine.const == 3.5 * np.ones((3, 5))).all()
    assert affine4.__repr__() == '3x5 bi-affine expressions'

    affine5 = x*z[:5] + yy[:, :1].T - 2.5*x + 3.5
    assert isinstance(affine5, ro.RoAffine)
    assert affine5.__repr__() == '1x5 bi-affine expressions'

    with pytest.raises(TypeError):
        z[-5:] * y - x

    with pytest.raises(TypeError):
        x * affine1

    with pytest.raises(TypeError):
        z[:5] * affine1

    coef_array = np.arange(1, 6)
    array1 = coef_array * affine1 + x
    assert isinstance(array1, ro.RoAffine)
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
    assert isinstance(array2, ro.RoAffine)
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
    assert isinstance(array3, ro.RoAffine)
    new_linear = (np.diag(coef_array) @ affine4.raffine[slice]).linear.toarray()
    new_const = (np.diag(coef_array) @ affine4.raffine[slice]).const
    assert (array3.raffine.linear.toarray() == new_linear).sum()
    assert (array3.raffine.const == new_const).sum()
    new_linear = (coef_array * affine4.affine[index] + x).linear.toarray()
    assert (array3.affine.linear.toarray() == new_linear).all()
    assert (array3.affine.const ==
            coef_array*affine4.affine.const[index] - 0.5).all()
    assert array3.__repr__() == '5 bi-affine expressions'


@pytest.mark.parametrize('array, const', [
    (rd.rand(), rd.rand()),
    (rd.rand(7), rd.rand()),
    (rd.rand(7), rd.rand(1)),
    (rd.rand(3, 7), rd.rand(7)),
    (rd.rand(3, 4), rd.rand(3, 4))
])
def test_ro_ldr_adapt(array, const):

    m = ro.Model()
    a = m.dvar()
    shape = array.shape if not isinstance(array, float) else ()
    y = m.ldr(shape)

    cshape = const.shape if not isinstance(const, float) else ()
    z = m.rvar(cshape)
    uset = (z >= 100, z <= 100)

    y.adapt(z)

    m.min(a)
    m.st(a >= 0)
    m.st((const - y == - array*z).forall(uset))

    with pytest.raises(RuntimeError):
        y.get()
    with pytest.raises(RuntimeError):
        y.get(z)

    m.solve(grb)

    y0 = y.get() * np.ones(cshape)
    yz = y.get(z).sum(axis=tuple(range(len(shape), len(shape+cshape))))

    with pytest.raises(ValueError):
        y.get(a)

    assert (abs(y0 - const*np.ones(y0.shape)) <= 1e-4).all()
    assert (abs(yz - array) <= 1e-4).all()


def test_ro_ldr_slice():

    model = ro.Model()

    x = model.ldr(5)
    xx = model.ldr((3, 4, 5))
    z = model.rvar(5)
    zz = model.rvar((4, 5))
    x.adapt(z)

    affine1 = x + xx[2, 1]
    assert affine1.shape == (5, )
    assert affine1.__repr__() == '5 bi-affine expressions'

    affine2 = affine1 - zz
    assert affine2.shape == (4, 5)
    assert affine2.__repr__() == '4x5 bi-affine expressions'

    affine3 = 5 + xx@rd.rand(5, 3)
    assert affine3.shape == (3, 4, 3)
    assert affine3.__repr__() == '3x4x3 affine expressions'

    affine4 = -(xx[0]@rd.rand(5, 3)) - xx[0, :, :3]
    assert affine4.shape == (4, 3)
    assert affine4.__repr__() == '4x3 affine expressions'

    affine5 = rd.rand(3, 4)@xx[0] + xx[:, 0, :]
    assert affine5.shape == (3, 5)
    assert affine5.__repr__() == '3x5 affine expressions'

    affine6 = 3 - xx[0] + xx[0]*rd.rand(5)
    assert affine6.shape == (4, 5)
    assert affine6.__repr__() == '4x5 affine expressions'

    assert (xx[0] == 1).__repr__() == '20 linear constraints'

    assert (xx[1] >= 1).__repr__() == '20 linear constraints'

    assert (xx[2] <= 1).__repr__() == '20 linear constraints'

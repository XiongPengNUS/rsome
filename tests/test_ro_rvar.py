from rsome import ro
import numpy as np
import pytest


def test_rvar_scalar():

    model = ro.Model()

    z = model.rvar()
    assert z.shape == ()
    assert z.ndim == 0
    assert z.vtype == 'C'
    assert z.__repr__() == 'a random variable'

    za = z.to_affine()
    assert za.shape == ()
    assert za.const == 0
    assert za.__repr__() == 'an affine expression'

    with pytest.raises(TypeError):
        model.rvar(vtype='d')

    with pytest.raises(TypeError):
        model.rvar(vtype='I')


def test_rvar_array():

    model = ro.Model()
    z = model.rvar(4)
    assert z.shape == (4, )
    assert z.ndim == 1
    assert z.vtype == 'C'
    assert z.__repr__() == '4 random variables'

    za = z.to_affine()
    assert za.shape == z.shape
    assert (za.const == np.zeros(z.shape)).all()
    assert za.__repr__() == '4 affine expressions'

    assert z[2].to_affine().shape == ()
    assert z[2].__repr__() == 'a random variable'
    assert z[2:4].to_affine().shape == (2,)
    assert z[2:4].__repr__() == '2 random variables'

    with pytest.raises(TypeError):
        model.rvar(4, vtype='B')

    with pytest.raises(TypeError):
        model.rvar(4.0)

    z = model.rvar((2, 3, 4))
    assert z.ndim == 3
    assert z.shape == (2, 3, 4)
    assert z.vtype == 'C'
    assert z.__repr__() == '2x3x4 random variables'

    za = z.to_affine()
    assert za.shape == z.shape
    assert (za.const == np.zeros(z.shape)).all()
    assert za.__repr__() == '2x3x4 affine expressions'

    assert z[:, 1].to_affine().shape == (2, 4)
    assert z[:, 1].__repr__() == '2x4 random variables'
    assert z[:, 1:3].to_affine().shape == (2, 2, 4)
    assert z[:, 1:3].__repr__() == '2x2x4 random variables'
    assert z[0, 1:3, :].to_affine().shape == (2, 4)
    assert z[0, 1:3].__repr__() == '2x4 random variables'
    assert z[0, 2, 1].to_affine().shape == ()
    assert z[0, 2, 1].__repr__() == 'a random variable'

    assert z.T.shape == (4, 3, 2)
    assert (z.T.const == np.zeros((4, 3, 2))).all()

    with pytest.raises(TypeError):
        model.rvar((2, 3, 4), vtype='d')

    with pytest.raises(TypeError):
        model.rvar((2, 3, 4), vtype='B')

    with pytest.raises(TypeError):
        model.rvar((2, 3.0, 4))

    z = model.rvar((2, 3, 4))
    with pytest.raises(IndexError):
        print(z[3])
    with pytest.raises(IndexError):
        print(z[:, 0, 2, 0])

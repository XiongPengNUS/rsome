from rsome import ro
import numpy as np
import pytest


def test_dvar_scalar():

    model = ro.Model()

    x = model.dvar(vtype='i', name='x')
    assert x.shape == ()
    assert x.ndim == 0
    assert x.vtype == 'I'
    assert x.__repr__() == 'x: an integer decision variable'

    xa = x.to_affine()
    assert xa.shape == ()
    assert xa.const == 0
    assert xa.__repr__() == 'an affine expression'

    with pytest.raises(ValueError):
        model.dvar(vtype='d')

    with pytest.raises(ValueError):
        model.dvar(vtype='BI')


def test_dvar_array():

    model = ro.Model()
    x = model.dvar(4, vtype='B', name='x')
    assert x.shape == (4, )
    assert x.ndim == 1
    assert x.vtype == 'B'
    assert x.__repr__() == 'x: 4 binary decision variables'

    xa = x.to_affine()
    assert xa.shape == x.shape
    assert (xa.const == np.zeros(x.shape)).all()
    assert xa.__repr__() == '4 affine expressions'

    assert x[2].to_affine().shape == ()
    assert x[2].__repr__() == 'slice of x: a binary decision variable'
    assert x[2:4].to_affine().shape == (2,)
    assert x[2:4].__repr__() == 'slice of x: 2 binary decision variables'

    with pytest.raises(ValueError):
        y = model.dvar(4, vtype='d')

    with pytest.raises(ValueError):
        y = model.dvar(4, vtype='BI')

    with pytest.raises(TypeError):
        y = model.dvar(4.0, vtype='B')

    y = model.dvar(5, vtype='BBICI')
    with pytest.raises(IndexError):
        print(y[5])
    with pytest.raises(IndexError):
        print(y[:, 0])

    x = model.dvar((2, 3, 4))
    assert x.ndim == 3
    assert x.shape == (2, 3, 4)
    assert x.vtype == 'C'
    assert x.__repr__() == '2x3x4 continuous decision variables'

    xa = x.to_affine()
    assert xa.shape == x.shape
    assert (xa.const == np.zeros(x.shape)).all()
    assert xa.__repr__() == '2x3x4 affine expressions'

    assert x[:, 1].to_affine().shape == (2, 4)
    assert x[:, 1].__repr__() == '2x4 continuous decision variables'
    assert x[:, 1:3].to_affine().shape == (2, 2, 4)
    assert x[:, 1:3].__repr__() == '2x2x4 continuous decision variables'
    assert x[0, 1:3, :].to_affine().shape == (2, 4)
    assert x[0, 1:3].__repr__() == '2x4 continuous decision variables'
    assert x[0, 2, 1].to_affine().shape == ()
    assert x[0, 2, 1].__repr__() == 'a continuous decision variable'

    assert x.T.shape == (4, 3, 2)
    assert (x.T.const == np.zeros((4, 3, 2))).all()
    assert x.T.__repr__() == '4x3x2 affine expressions'

    with pytest.raises(ValueError):
        y = model.dvar((2, 3, 4), vtype='d')

    with pytest.raises(ValueError):
        y = model.dvar((2, 3, 4), vtype='BI')

    with pytest.raises(TypeError):
        y = model.dvar((2, 3.0, 4), vtype='B')

    y = model.dvar((2, 3, 4))
    with pytest.raises(IndexError):
        print(y[3])
    with pytest.raises(IndexError):
        print(y[:, 0, 2, 0])

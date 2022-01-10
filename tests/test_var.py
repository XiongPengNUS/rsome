from src.rsome import ro


def test_dvar():
    model = ro.Model()

    x = model.dvar(vtype='I')
    assert x.shape == (1, )
    assert x.vtype == 'I'

    x = model.dvar(4, vtype='B')
    assert x.shape == (4, )
    assert x.vtype == 'B'

    x = model.dvar((2, 3, 4))
    assert x.shape == (2, 3, 4)
    assert x.vtype == 'C'
    assert x[:, 1].to_affine().shape == (2, 4)
    assert x[0, 2, 1].to_affine().shape == (1, )
    assert x.T.shape == (4, 3, 2)

import rsome as rso
from rsome import dro
from rsome import ro
from rsome import grb_solver as grb
from rsome import eco_solver as eco
from rsome import E
import numpy as np
import numpy.random as rd
import pytest


def test_model():

    model = dro.Model()

    x = model.dvar(3)
    z = model.rvar(3)
    fset = model.ambiguity()
    fset.suppset(z == 0)

    affine1 = z@x
    assert isinstance(affine1, rso.lp.DecRoAffine)
    assert affine1.__repr__() == 'a bi-affine expression'

    affine2 = z + z@x
    assert isinstance(affine2, rso.lp.DecRoAffine)
    assert affine2.__repr__() == '3 bi-affine expressions'

    affine3 = z@(x + 3)
    assert isinstance(affine3, rso.lp.DecRoAffine)
    assert affine3.__repr__() == 'a bi-affine expression'

    affine4 = affine2.sum()
    assert isinstance(affine4, rso.lp.DecRoAffine)
    assert affine4.__repr__() == 'a bi-affine expression'

    with pytest.raises(ValueError):
        model.min(x)

    with pytest.raises(ValueError):
        model.max(x)

    with pytest.raises(ValueError):
        model.minsup(x, fset)

    with pytest.raises(ValueError):
        model.maxinf(x, fset)

    c = rd.randn(3)
    model.max(c@x)
    model.st(rso.norm(x) <= 1)

    x_sol = c / (c**2).sum()**0.5
    objval = c @ x_sol

    with pytest.raises(RuntimeError):
        x.get()

    model.do_math()
    primal = model.do_math()

    primal_sol = primal.solve(grb)
    assert abs(primal_sol.objval + objval) < 1e-4
    assert (abs(primal_sol.x[2:5] - x_sol) < 1e-4).all()

    model.do_math(primal=False)
    dual = model.do_math(primal=False)

    dual_sol = dual.solve(grb)
    assert abs(dual_sol.objval - objval) < 1e-4

    with pytest.raises(SyntaxError):
        model.min(x.sum())

    with pytest.raises(SyntaxError):
        model.max(x.sum())

    with pytest.raises(SyntaxError):
        model.minsup(x.sum(), fset)

    with pytest.raises(SyntaxError):
        model.maxinf(x.sum(), fset)

    model.st(x - abs(c)*10 >= 0)
    with pytest.warns(UserWarning):
        model.solve(grb)

    with pytest.raises(RuntimeError):
        objval = model.get()

    with pytest.raises(RuntimeError):
        x_sol = x.get()


@pytest.mark.parametrize('array, r', [
    (rd.rand(3, 5), 0.02),
    (rd.rand(5, 2), 0.01),
    (rd.rand(9, 3), 0.001),
])
def test_kl_prob(array, r):

    ns, n = array.shape
    m1 = dro.Model(ns)
    x = m1.dvar(n)

    z = m1.rvar(n)
    fset = m1.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    pr = m1.p
    fset.probset(pr.kldiv(1/ns, r))

    m1.maxinf(E(z @ x), fset)
    m1.st(x == 1)
    m1.solve(eco)

    m2 = ro.Model()
    p = m2.dvar(ns)

    values = array.sum(axis=1)
    m2.min(p @ values)
    m2.st(p >= 0, p.sum() == 1)
    m2.st(p.kldiv(1/ns, r))
    m2.solve(eco)

    assert abs(m1.get() - m2.get()) < 1e-5


def test_model_match():

    m1, m2 = dro.Model(name='1st model'), dro.Model(name='2nd model')

    x1, x2 = m1.dvar(5), m2.dvar(5)
    xx1, xx2 = m1.dvar((2, 5)), m2.dvar((2, 5))

    y1, y2 = m1.dvar(5), m2.dvar(5)
    yy1, yy2 = m1.dvar((2, 5)), m2.dvar((2, 5))

    z1, z2 = m1.rvar(5), m2.rvar(5)
    zz1, zz2 = m1.rvar((2, 5)), m2.rvar((2, 5))

    fset1, fset2 = m1.ambiguity(), m2.ambiguity()
    p1, p2 = m1.p, m2.p
    fset1.suppset(z1 == 0, zz1 == 1)
    fset1.exptset(E(z1) == 0, E(zz1) == 1)
    fset2.suppset(z2 == 0, zz2 == 1)
    fset2.exptset(E(z2) == 0, E(zz2) == 1)
    fset = m1.ambiguity()
    with pytest.raises(ValueError):
        fset.suppset(z1 == 0, zz2 == 1)
    with pytest.raises(ValueError):
        fset1.probset(p2 == 1)

    y1.adapt(z1)
    y2.adapt(z2)
    yy1.adapt(zz1)
    yy2.adapt(zz2)

    with pytest.raises(ValueError):
        y1.adapt(z2)

    with pytest.raises(TypeError):
        (y1 + 3) @ z1

    with pytest.raises(ValueError):
        x1 + x2

    with pytest.raises(ValueError):
        x1 + xx2

    with pytest.raises(ValueError):
        p1 + p2

    with pytest.raises(ValueError):
        x1 + y2

    with pytest.raises(ValueError):
        xx1 + y2

    with pytest.raises(ValueError):
        x1 + z2

    with pytest.raises(ValueError):
        xx1 + z2

    with pytest.raises(ValueError):
        y1 + z2

    with pytest.raises(ValueError):
        yy1 + zz2

    with pytest.raises(ValueError):
        xx1*z2

    with pytest.raises(ValueError):
        xx2*z1

    with pytest.raises(ValueError):
        xx1*z1 + xx2*z2

    with pytest.raises(ValueError):
        (xx1*z1 <= 0).forall(fset2)

    with pytest.raises(ValueError):
        (xx1*z1 <= 0).forall(z2 == 0)

    with pytest.raises(ValueError):
        (xx1*z1 <= 0).forall((z2 >= 0, ))

    with pytest.raises(ValueError):
        (xx1*z1 <= 0).forall(abs(z2 - 1) <= 0.1)

    with pytest.raises(ValueError):
        x1@z2

    with pytest.raises(ValueError):
        xx1@z2

    with pytest.raises(ValueError):
        m1.st(xx2 + 0 <= 1)

    with pytest.raises(ValueError):
        m1.st(x2 >= 0)

    with pytest.raises(ValueError):
        m1.st(x2 + z2 >= 0)

    with pytest.raises(ValueError):
        m1.st(z2*x2 + xx2 >= 0)

    with pytest.raises(TypeError):
        m1.st(np.zeros(5) == 0)

    with pytest.raises(TypeError):
        m1.st(z2*x2 + abs(x2) >= 0)

    with pytest.raises(TypeError):
        z1@x1 + E(z1@x1)

    with pytest.raises(TypeError):
        E(z2@x2) + z2@x2

    with pytest.raises(ValueError):
        rso.norm(y1)

    with pytest.raises(ValueError):
        rso.square(y1)

    with pytest.raises(ValueError):
        rso.sumsqr(y1)

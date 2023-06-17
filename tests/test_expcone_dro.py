import rsome as rso
from rsome import dro
from rsome import E
from rsome import eco_solver as eco
import numpy as np


def test_single_dro():

    n = 8
    array = np.random.rand(n)
    pr = np.random.rand()

    m = dro.Model()

    z = m.rvar(n)
    u = m.rvar()
    v = m.rvar()

    uset = (rso.norm(z) <= u,
            u <= rso.entropy(v),
            v == pr)
    fset = m.ambiguity()
    fset.suppset(uset)

    x = m.dvar(n)
    m.maxinf(x@z, fset)
    m.st(x == array)
    m.solve(eco)

    uvalue = -pr*np.log(pr)
    z_sol = - array / (array**2).sum()**0.5 * uvalue
    objval = array @ z_sol

    assert abs(objval - m.get()) < 1e-4


def test_multiple_dro():

    n = 2
    ns = 5
    array = np.random.rand(ns, n)
    pr = np.random.rand()

    m = dro.Model()

    z = m.rvar(n)
    u = m.rvar()
    v = m.rvar()

    uset = (rso.norm(z) <= u,
            u <= rso.entropy(v),
            v == pr)

    x = m.dvar(ns)
    m.min(x.sum())
    m.st((x >= array@z).forall(uset))
    m.solve(eco)

    uvalue = -pr*np.log(pr)
    objval = ((array * array).sum(axis=1) ** 0.5 * uvalue).sum()

    assert abs(objval - m.get()) < 1e-4


def test_sw_cone():
    n = 8
    s = 4
    array = np.random.rand(n)
    pr = np.random.rand(s)
    ctr = np.random.rand(s, n)

    m = dro.Model(s)
    z = m.rvar(n)
    u = m.rvar()
    v = m.rvar()

    fset = m.ambiguity()
    for j in range(s):
        fset[j].suppset(rso.norm(z - ctr[j]) <= u,
                        u <= rso.entropy(v),
                        v == pr[j])

    p = m.p
    fset.probset(p == 1/s)

    x = m.dvar(n)
    y = m.dvar()
    for j in range(s):
        y.adapt(j)
    m.maxinf(E(y), fset)
    m.st(y <= x@z)
    m.st(x == array)
    m.solve(eco)

    uvalues = -pr*np.log(pr)
    objval = ctr@array - ((array**2).sum()**0.5) * uvalues

    assert abs(objval.mean() - m.get()) < 1e-4

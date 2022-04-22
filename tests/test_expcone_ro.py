import rsome as rso
from rsome import ro
from rsome import eco_solver as eco
import numpy as np


def test_single_ro():

    n = 8
    array = np.random.rand(n)
    pr = np.random.rand()

    m = ro.Model()

    z = m.rvar(n)
    u = m.rvar()
    v = m.rvar()

    uset = (rso.norm(z) <= u,
            u <= rso.entropy(v),
            v == pr)

    x = m.dvar(n)
    m.maxmin(x@z, uset)
    m.st(x == array)
    m.solve(eco)

    uvalue = -pr*np.log(pr)
    z_sol = - array / (array**2).sum()**0.5 * uvalue
    objval = array @ z_sol

    assert abs(objval - m.get()) < 1e-4


def test_multiple_ro():

    n = 2
    ns = 5
    array = np.random.rand(ns, n)
    pr = np.random.rand()

    m = ro.Model()

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

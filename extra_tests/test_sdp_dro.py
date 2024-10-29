import rsome as rso
from rsome import dro
from rsome import E
from rsome import msk_solver as msk
from rsome import cpt_solver as cpt
import numpy as np
import pandas as pd


def test_log_det_dro():

    n = 4
    rv = np.random.rand(n, n)
    A = rv@rv.T + 0.5

    m = dro.Model()

    X = m.dvar((n, n))
    Z = rso.tril(m.dvar((n, n)))
    v = m.dvar(n)

    m.max(v.sum())
    m.st(rso.rstack([X, Z],
                    [Z.T, rso.diag(Z, fill=True)]) >> 0)
    m.st(v <= rso.log(rso.diag(Z)))
    m.st(X == A)

    m.solve(msk)

    objval = np.linalg.det(A)
    assert abs(objval - np.exp(m.get())) < 1e-5

    m.solve(cpt)

    objval = np.linalg.det(A)
    assert abs(objval - np.exp(m.get())) < 1e-5


def test_sw_log_det_dro():

    n = 4
    s = 3

    As = []
    for j in range(s):
        rv = np.random.rand(n, n)
        A = rv@rv.T + 0.5
        As.append(A)

    m = dro.Model(s)
    Z = m.rvar((n, n))

    fset = m.ambiguity()
    for j in range(s):
        fset[j].suppset(Z == As[j])
    p = m.p
    fset.probset(p == 1/s)

    X = m.dvar((n, n))
    Yfull = m.dvar((n, n))
    Y = rso.tril(Yfull)
    v = m.dvar(n)
    for j in range(s):
        X.adapt(j)
        Yfull.adapt(j)
        v.adapt(j)

    m.maxinf(E(v.sum()), fset)
    m.st(rso.rstack([X, Y],
                    [Y.T, rso.diag(Y, fill=True)]) >> 0)
    m.st(v <= rso.log(rso.diag(Y)))
    m.st(X == Z)

    m.solve(msk)

    assert all(abs(np.exp(v.sum()()) -
                   pd.Series([np.linalg.det(A) for A in As])) < 1e-6)

    objval = np.log(np.array([np.linalg.det(A) for A in As])).mean()
    assert abs(objval - m.get()) < 1e-5

    m.solve(cpt)

    assert all(abs(np.exp(v.sum()()) -
                   pd.Series([np.linalg.det(A) for A in As])) < 1e-6)

    objval = np.log(np.array([np.linalg.det(A) for A in As])).mean()
    assert abs(objval - m.get()) < 1e-5

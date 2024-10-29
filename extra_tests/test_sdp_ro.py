import rsome as rso
from rsome import ro
from rsome import msk_solver as msk
from rsome import cpt_solver as cpt
import numpy as np


def test_log_det_ro():

    n = 4
    rv = np.random.rand(n, n)
    A = rv@rv.T + 0.5

    m = ro.Model()

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
    assert abs(objval - np.exp(m.get())) < 1e-5

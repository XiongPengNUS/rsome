from rsome import dro
from rsome import cpx_solver as cpx
from rsome import E
import rsome as rso
import numpy as np
import numpy.random as rd


def test_dro_model():

    rd.seed(5)

    N = 3
    S = 20
    c = np.ones(N)
    d = 50 * N
    p = 1 + 4*rd.rand(N)
    zbar = 100 * rd.rand(N)
    zhat = zbar * rd.rand(S, N)
    theta = 0.01 * zbar.min()

    model = dro.Model(S)
    z = model.rvar(N)
    u = model.rvar()

    fset = model.ambiguity()
    for s in range(S):
        fset[s].suppset(0 <= z, z <= zbar,
                        rso.norm(z-zhat[s]) <= u)
    fset.exptset(E(u) <= theta)
    pr = model.p
    fset.probset(pr == 1/S)

    x = model.dvar(N)
    y = model.dvar(N)
    y.adapt(z)
    y.adapt(u)
    for s in range(S):
        y.adapt(s)

    model.minsup(-p@x + E(p@y), fset)
    model.st(y >= 0)
    model.st(y >= x - z)
    model.st(x >= 0)
    model.st(c@x == d)

    model.solve(cpx)

    objval = -229.1826
    x_sol = np.array([65.79278894, 45.06811655, 39.13909451])

    assert abs(model.get() - objval) < 1e-4
    assert (abs(x_sol - x.get()) < 1e-4).all()
    assert len(y.get()) == S
    assert len(y.get(z)) == S
    assert len(y.get(u)) == S

    dual_sol = model.do_math(primal=False).solve(cpx)
    assert abs(dual_sol.objval + objval) < 1e-4

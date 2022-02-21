from rsome import ro
from rsome import dro
from rsome import cpx_solver as cpx
import rsome as rso
import numpy as np
import pytest


def test_ro_model():

    n = 150
    step = 1
    i = np.arange(1, n+1, step)
    p = 1.15 + 0.05/n*i
    delta = 0.05/450 * (2*i*n*(n+1))**0.5
    Gamma = 5

    model = ro.Model()
    x = model.dvar(n)
    z = model.rvar(n)

    model.maxmin((p + delta*z) @ x,
                 rso.norm(z, np.infty) <= 1,
                 rso.norm(z, 1) <= Gamma)
    model.st(sum(x) == 1)
    model.st(x >= 0)

    model.solve(cpx)
    model.get()

    objval = 1.17089

    assert abs(model.get() - objval) < 1e-4

    primal_sol = model.do_math().solve(cpx)
    assert abs(primal_sol.objval + objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(cpx)
    assert abs(dual_sol.objval - objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(cpx)
    assert abs(dual_sol.objval - objval) < 1e-4

    with pytest.raises(TypeError):
        z.get()


def test_dro_model():

    n = 150
    step = 1
    i = np.arange(1, n+1, step)
    p = 1.15 + 0.05/n*i
    delta = 0.05/450 * (2*i*n*(n+1))**0.5
    # sig = 0.05/450 * (2*i*n*(n+1))**0.5
    Gamma = 5

    model = dro.Model()
    x = model.dvar(n)
    z = model.rvar(n)
    fset = model.ambiguity()
    fset.suppset(rso.norm(z, np.infty) <= 1,
                 rso.norm(z, 1) <= Gamma)

    model.maxinf((p + delta*z) @ x, fset)
    model.st(sum(x) == 1)
    model.st(x >= 0)

    model.solve(cpx)
    model.get()

    objval = 1.17089

    assert abs(model.get() - objval) < 1e-4

    primal_sol = model.do_math().solve(cpx)
    assert abs(primal_sol.objval + objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(cpx)
    assert abs(dual_sol.objval - objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(cpx)
    assert abs(dual_sol.objval - objval) < 1e-4

    with pytest.raises(TypeError):
        z.get()

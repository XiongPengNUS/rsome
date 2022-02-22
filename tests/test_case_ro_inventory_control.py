from rsome import ro
from rsome import dro
from rsome import ort_solver as ort
import numpy as np


def test_ro_model():

    T = 24
    t = np.arange(1, T+1)
    d0 = 1000 * (1 + 0.5*np.sin(np.pi*(t-1)/12))
    alpha = np.array([1, 1.5, 2]).reshape((3, 1))
    c = alpha * (1 + 0.5*np.sin(np.pi*(t-1)/12))

    P = 567
    Q = 13600
    vmin = 500
    vmax = 2000
    v = 500
    theta = 0.2

    model = ro.Model()
    d = model.rvar(T)
    uset = (d >= (1-theta)*d0, d <= (1+theta)*d0)

    p = model.ldr((3, T))
    for t in range(1, T):
        p[:, t].adapt(d[:t])

    model.minmax((c*p).sum(), uset)
    model.st(0 <= p, p <= P)
    model.st(p.sum(axis=1) <= Q)
    model.st(v + p[:, :t+1].sum() - d[:t+1].sum() >= vmin for t in range(T))
    model.st(v + p[:, :t+1].sum() - d[:t+1].sum() <= vmax for t in range(T))

    model.solve(ort)

    objval = 44272.82749
    assert abs(objval - model.get()) < 1e-4

    primal_sol = model.do_math().solve(ort)
    assert abs(primal_sol.objval - objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(ort)
    assert abs(dual_sol.objval + objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(ort)
    assert abs(dual_sol.objval + objval) < 1e-4


def test_dro_model():

    T = 24
    t = np.arange(1, T+1)
    d0 = 1000 * (1 + 0.5*np.sin(np.pi*(t-1)/12))
    alpha = np.array([1, 1.5, 2]).reshape((3, 1))
    c = alpha * (1 + 0.5*np.sin(np.pi*(t-1)/12))

    P = 567
    Q = 13600
    vmin = 500
    vmax = 2000
    v = 500
    theta = 0.2

    model = dro.Model()
    d = model.rvar(T)
    fset = model.ambiguity()
    fset.suppset(d >= (1-theta)*d0, d <= (1+theta)*d0)

    p = model.dvar((3, T))
    for t in range(1, T):
        p[:, t].adapt(d[:t])

    model.minsup((c*p).sum(), fset)
    model.st(0 <= p, p <= P)
    model.st(p.sum(axis=1) <= Q)
    model.st(v + p[:, :t+1].sum() - d[:t+1].sum() >= vmin for t in range(T))
    model.st(v + p[:, :t+1].sum() - d[:t+1].sum() <= vmax for t in range(T))

    model.solve(ort)

    objval = 44272.82749
    assert abs(objval - model.get()) < 1e-4

    primal_sol = model.do_math().solve(ort)
    assert abs(primal_sol.objval - objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(ort)
    assert abs(dual_sol.objval + objval) < 1e-4

    dual_sol = model.do_math(primal=False).solve(ort)
    assert abs(dual_sol.objval + objval) < 1e-4

    assert p.get().shape == (3, T)
    assert p.get(d).shape == (3, T, T)

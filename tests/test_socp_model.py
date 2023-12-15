from rsome import socp
from rsome import grb_solver as grb
import rsome as rso
import numpy as np
import numpy.random as rd
import pandas as pd
import pytest


def test_milp_model():

    model = socp.Model()
    x = model.dvar()
    y = model.dvar(vtype='I')
    z = model.dvar(2)

    with pytest.raises(ValueError):
        model.min(z*5)

    with pytest.raises(ValueError):
        model.max(z*5)

    model.max(3*x + 4*y)
    model.st(x*2.5 + y <= 20)
    model.st(-5*x - 3*y >= -30)
    model.st(-16 + x + 2*y <= 0)
    model.st(abs(y) <= 2)

    with pytest.raises(RuntimeError):
        model.get()

    with pytest.raises(RuntimeError):
        x.get()

    primal = model.do_math()
    df = primal.show()
    assert isinstance(df, pd.DataFrame)

    model.solve(grb)
    assert abs(model.get() - 22.4) < 1e-4
    assert abs(x.get() - 4.8) < 1e-4
    assert abs(y.get() - 2) < 1e-4
    assert primal == model.do_math()

    model = socp.Model()
    x = model.dvar()
    y = model.dvar(vtype='I')
    z = model.dvar(2)
    model.min(-3*x + -4*y)
    model.st(x*2.5 + y <= 20)
    model.st(-5*x - 3*y >= -30)
    model.st(-16 + x + 2*y <= 0)
    model.st(abs(y) <= 2)
    model.solve(grb)
    assert abs(model.get() + 22.4) < 1e-4
    assert abs(x.get() - 4.8) < 1e-4
    assert abs(y.get() - 2) < 1e-4

    with pytest.warns(UserWarning):
        model.do_math(primal=False)

    model.st(y >= 2.5)
    with pytest.warns(UserWarning):
        model.solve(grb)

    with pytest.raises(RuntimeError):
        objval = model.get()
        print(objval)

    with pytest.raises(RuntimeError):
        x_sol = x.get()
        print(x_sol)


def test_socp_model():

    n = 10
    c = rd.randn(n)

    model = socp.Model()
    x = model.dvar(n)
    model.max(c@x)
    model.st(-rso.norm(x) >= -1)

    x_sol = c / (c**2).sum()**0.5
    objval = c @ x_sol

    with pytest.raises(RuntimeError):
        model.get()

    with pytest.raises(RuntimeError):
        x.get()

    model.solve(grb)
    assert abs(model.get() - objval) < 1e-6
    assert (abs(x_sol - x.get()) < 1e-3).all()

    dual = model.do_math(primal=False)
    dual_sol = dual.solve(grb)
    assert abs(dual_sol.objval - objval) < 1e-6


def test_model_match():

    m1, m2 = socp.Model('1st model'), socp.Model('2nd model')

    x1, x2 = m1.dvar(5), m2.dvar(5)
    xx1, xx2 = m1.dvar((2, 5)), m2.dvar((2, 5))

    with pytest.raises(ValueError):
        x1 + x2

    with pytest.raises(ValueError):
        x1 + xx2

    with pytest.raises(ValueError):
        m2.st(xx1 + 0 <= 1)

    with pytest.raises(ValueError):
        m1.st(x2 >= 0)

    with pytest.raises(TypeError):
        m1.st(np.zeros(5) == 0)

    with pytest.raises(ValueError):
        m1.st(rso.norm(x2) <= 1)

    with pytest.raises(TypeError):
        rso.square(x1) + abs(x2)

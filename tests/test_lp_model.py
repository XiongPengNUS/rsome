from rsome import lp
from rsome import grb_solver as grb
import rsome as rso
import numpy as np
import pandas as pd
import pytest


def test_model():

    model = lp.Model()
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
    model.st(y.abs() <= 2)

    with pytest.raises(RuntimeError):
        model.get()

    with pytest.raises(RuntimeError):
        x.get()

    primal = model.do_math()
    string = '=============================================\n'
    string += 'Number of variables:           5\n'
    string += 'Continuous/binaries/integers:  4/0/1\n'
    string += '---------------------------------------------\n'
    string += 'Number of linear constraints:  6\n'
    string += 'Inequalities/equalities:       6/0\n'
    string += 'Number of coefficients:        11\n'
    assert primal.__repr__() == string

    df = primal.show()
    assert isinstance(df, pd.DataFrame)

    model.solve(grb)
    print(model.solution.status)
    assert abs(model.get() - 22.4) < 1e-4
    assert abs(x.get() - 4.8) < 1e-4
    assert abs(y.get() - 2) < 1e-4
    assert primal == model.do_math()

    model = lp.Model()
    x = model.dvar()
    y = model.dvar(vtype='I')
    z = model.dvar(2)
    model.min(-3*x + -4*y)
    model.st(x*2.5 + y <= 20)
    model.st(-5*x - 3*y >= -30)
    model.st(-16 + x + 2*y <= 0)
    model.st((y*2).abs() <= 4)
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

    with pytest.raises(SyntaxError):
        model.min(3*x + 5*y)

    with pytest.raises(SyntaxError):
        model.max(3*x + 5*y)


def test_model_match():

    m1, m2 = lp.Model('1st model'), lp.Model('2nd model')

    x1, x2 = m1.dvar(5), m2.dvar(5)
    xx1, xx2 = m1.dvar((2, 5)), m2.dvar((2, 5))

    with pytest.raises(ValueError):
        x1 + x2

    with pytest.raises(ValueError):
        x1 + xx2

    with pytest.raises(ValueError):
        m1.st(xx2 + 0 <= 1)

    with pytest.raises(ValueError):
        m2.st(xx1 + 0 >= 1)

    with pytest.raises(ValueError):
        m1.st(x2 >= 0)

    with pytest.raises(TypeError):
        m1.st(np.zeros(5) == 0)

    with pytest.raises(TypeError):
        m1.st(rso.norm(x1) <= 1)

import rsome as rso
from rsome import ro
from rsome import grb_solver as grb
import numpy as np
import pandas as pd
import numpy.random as rd
import pytest


def test_model():

    model = ro.Model()

    x = model.dvar(3)
    z = model.rvar()

    with pytest.raises(ValueError):
        model.min(x)

    with pytest.raises(ValueError):
        model.max(x)

    with pytest.raises(ValueError):
        model.minmax(x, z == 0)

    with pytest.raises(ValueError):
        model.maxmin(x, z == 0)

    c = rd.randn(3)
    model.max(c@x)
    model.st(rso.norm(x) <= 1)

    x_sol = c / (c**2).sum()**0.5
    objval = c @ x_sol

    with pytest.raises(RuntimeError):
        x.get()

    primal = model.do_math()
    string = 'Conic program object:\n'
    string += '=============================================\n'
    string += 'Number of variables:           8\n'
    string += 'Continuous/binaries/integers:  8/0/0\n'
    string += '---------------------------------------------\n'
    string += 'Number of linear constraints:  5\n'
    string += 'Inequalities/equalities:       2/3\n'
    string += 'Number of coefficients:        11\n'
    string += '---------------------------------------------\n'
    string += 'Number of SOC constraints:     1\n'
    string += '---------------------------------------------\n'
    string += 'Number of ExpCone constraints: 0\n'
    assert primal.__repr__() == string

    df = primal.show()
    assert type(df) == pd.DataFrame
    assert (df.loc['Obj', 'x1':'x8'].values == np.array([1] + [0]*7)).all()
    eye = np.eye(3)
    rows = np.concatenate((np.zeros((3, 1)), eye, -eye, np.zeros((3, 1))), axis=1)
    assert (df.loc['LC1':'LC3', 'x1':'x8'].values == rows).all()
    assert (df.loc['LC1':'LC3', 'sense'].values == '==').all()
    assert (df.loc['LC1':'LC3', 'constant'].values == 0).all()
    assert (df.loc['LC4', 'x1':'x8'].values == np.array([0]*7 + [1])).all()
    assert df.loc['LC4', 'sense'] == '<='
    assert df.loc['LC4', 'constant'] == 1.0
    rows = np.zeros(8)
    rows[0], rows[1:4] = -1, -c
    assert (df.loc['LC5', 'x1':'x8'].values == rows).all()
    assert df.loc['LC5', 'sense'] == '<='
    assert df.loc['LC5', 'constant'] == 0.0
    assert (df.loc['QC1', 'x1':'x8'].values == np.array([0]*4 + [1]*3 + [-1])).all()
    assert df.loc['QC1', 'sense'] == '<='
    assert df.loc['QC1', 'constant'] == 0.0
    ub = np.array([np.inf] * 8)
    assert (df.loc['UB', 'x1':'x8'].values == ub).all()
    lb = np.array([-np.inf] * 8)
    lb[-1] = 0
    assert (df.loc['LB', 'x1':'x8'].values == lb).all()
    assert (df.loc['Type', 'x1':'x8'].values == 'C').all()

    primal_sol = primal.solve(grb)
    assert abs(primal_sol.objval + objval) < 1e-4
    assert (abs(primal_sol.x[1:4] - x_sol) < 1e-4).all()

    dual = model.do_math(primal=False)
    string = 'Conic program object:\n'
    string += '=============================================\n'
    string += 'Number of variables:           5\n'
    string += 'Continuous/binaries/integers:  5/0/0\n'
    string += '---------------------------------------------\n'
    string += 'Number of linear constraints:  4\n'
    string += 'Inequalities/equalities:       0/4\n'
    string += 'Number of coefficients:        7\n'
    string += '---------------------------------------------\n'
    string += 'Number of SOC constraints:     1\n'
    string += '---------------------------------------------\n'
    string += 'Number of ExpCone constraints: 0\n'
    assert dual.__repr__() == string

    df = dual.show()
    assert type(df) == pd.DataFrame
    assert (df.loc['Obj', 'x1':'x5'].values == np.array([0]*3 + [1] + [0])).all()
    rows = np.zeros((4, 5))
    rows[0, -1], rows[1:, 0:3], rows[1:, -1] = -1, np.eye(3), -c
    assert (df.loc['LC1':'LC4', 'x1':'x5'].values == rows).all()
    assert (df.loc['LC1':'LC4', 'sense'].values == '==').all()
    assert (df.loc['LC1':'LC4', 'constant'].values == np.array([1] + [0]*3)).all()
    assert (df.loc['QC1', 'x1':'x5'].values == np.array([1, 1, 1, -1, 0])).all()
    assert (df.loc['UB', 'x1':'x5'].values == np.array([np.inf]*4 + [0])).all()
    lb = np.array([-np.inf]*5)
    lb[-2] = 0
    assert (df.loc['LB', 'x1':'x5'].values == lb).all()
    assert (df.loc['Type', 'x1':'x5'].values == 'C').all()
    assert (df == model.do_math(primal=False).show()).all().all()

    dual_sol = dual.solve(grb)
    assert abs(dual_sol.objval - objval) < 1e-4

    with pytest.raises(SyntaxError):
        model.min(x.sum())

    with pytest.raises(SyntaxError):
        model.max(x.sum())

    with pytest.raises(SyntaxError):
        model.minmax(x.sum(), z == 0)

    with pytest.raises(SyntaxError):
        model.maxmin(x.sum(), z == 0)

    model.st(x - abs(c)*10 >= 0)
    with pytest.warns(UserWarning):
        model.solve(grb)

    with pytest.raises(RuntimeError):
        objval = model.get()

    with pytest.raises(RuntimeError):
        x_sol = x.get()


def test_model_ro():

    n = 5
    array = rd.randn(n)

    model = ro.Model()
    x = model.dvar(n)
    z = model.rvar(n)
    model.min(z@x)
    model.st(rso.norm(x) <= 1)
    with pytest.raises(RuntimeError):
        model.solve(grb)

    model = ro.Model()
    x = model.dvar(n)
    z = model.rvar(n)
    model.minmax(z @ x, z == array)
    model.st(rso.norm(x) <= 1)
    model.solve(grb)

    x_sol = - array / (array**2).sum()**0.5
    objval = array @ x_sol

    assert abs(objval - model.get()) < 1e-4
    assert (abs(x_sol - x.get()) < 1e-4).all()

    with pytest.raises(SyntaxError):
        model.maxmin(z @ x, z == array)


def test_model_match():

    m1, m2 = ro.Model('1st model'), ro.Model('2nd model')

    x1, x2 = m1.dvar(5), m2.dvar(5)
    xx1, xx2 = m1.dvar((2, 5)), m2.dvar((2, 5))

    y1, y2 = m1.ldr(5), m2.ldr(5)
    yy1, yy2 = m1.ldr((2, 5)), m2.ldr((2, 5))

    z1, z2 = m1.rvar(5), m2.rvar(5)
    zz1, zz2 = m1.rvar((2, 5)), m2.rvar((2, 5))

    with pytest.raises(ValueError):
        y1.adapt(z2)

    with pytest.raises(ValueError):
        x1 + x2

    with pytest.raises(ValueError):
        x1 + xx2

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
        yy2 + zz1

    with pytest.raises(ValueError):
        xx1*z2

    with pytest.raises(ValueError):
        xx2*z1

    with pytest.raises(ValueError):
        xx1*z1 + xx2*z2

    with pytest.raises(ValueError):
        (xx1*z1 + 2) + (xx2*z2 - 1.5)

    with pytest.raises(ValueError):
        (xx1*z1 <= 0).forall(z1 == 1, z2 == 0)

    with pytest.raises(ValueError):
        x1@z2

    with pytest.raises(ValueError):
        xx1@z2

    with pytest.raises(ValueError):
        zz1@x2

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

    with pytest.raises(ValueError):
        m1.minmax(z1 @ x1, z2 == 0)

    with pytest.raises(ValueError):
        m1.maxmin(z1 @ x1, z2 == 0)

import rsome as rso
from rsome import lp
from rsome import socp
from rsome import ro
from rsome import lpg_solver as lpg
from rsome import ort_solver as ort
from rsome import eco_solver as eco
from rsome import grb_solver as grb
from rsome import msk_solver as msk
# from rsome import cpx_solver as cpx
from rsome import cpt_solver as cpt
import numpy as np
import numpy.random as rd
import gurobipy as gp
from mosek.fusion import ParameterError
import pytest


def test_lp():

    model = lp.Model()
    x = model.dvar()
    y = model.dvar()

    model.max(3*x + 4*y)
    model.st(2.5*x + y <= 20)
    model.st(-5*x - 3*y >= -30)
    model.st(x + 2*y <= 16)
    model.st(abs(y) <= 2)
    assert not model.optimal()

    model.solve()
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()

    model.solve(lpg)
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()

    """
    model.solve(clp)
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()
    """

    model.solve(ort)
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()

    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()

    model.solve(eco)
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()

    model.solve(grb, params={'TimeLimit': 100, 'FeasibilityTol': 1e-9})
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()
    model.do_math().to_lp('lp_test')
    gmodel = gp.read('lp_test.lp')
    gmodel.optimize()
    assert model.solution.objval == gmodel.objVal
    with pytest.raises(AttributeError):
        model.solve(grb, params={'NotAParameter': 1})

    model.solve(msk, params={'optimizerMaxTime': 100.0,
                             'simplexAbsTolPiv': 1e-9})
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()
    with pytest.raises(ParameterError):
        model.solve(msk, params={'not_a_parameter': 1})

    # model.solve(cpx, params={'timelimit': 100.0,
    #                          'simplex.tolerances.feasibility': 1e-7})
    # assert abs(model.get() - 22.4) < 1e-6
    # assert abs(x.get() - 4.8) < 1e-6
    # assert abs(y.get() - 2) < 1e-6
    # assert model.optimal()
    # with pytest.raises(ValueError):
    #     model.solve(cpx, params={'not_a_parameter': 1})

    model.solve(cpt)
    assert abs(model.get() - 22.4) < 1e-6
    assert abs(x.get() - 4.8) < 1e-6
    assert abs(y.get() - 2) < 1e-6
    assert model.optimal()


def test_mip():

    n = 10
    c = rd.rand(n)
    k = 4.8
    x_sol = np.zeros(n)
    # x_sol[c.argsort()[-int(k):]] = 1
    x_sol[c.argsort()[int(-k):]] = 1

    model = lp.Model()
    x = model.dvar(n, vtype='B')
    model.max(c@x)
    model.st(sum(x) <= k)
    model.st(x >= -10)
    model.st(x <= 10)

    model.solve()
    assert (x_sol == x.get().round()).all()
    assert model.optimal()

    model.solve(ort)
    assert (x_sol == x.get().round()).all()
    assert model.optimal()

    model.solve(eco)
    assert (x_sol == x.get().round()).all()
    assert model.optimal()

    model.solve(grb)
    assert (x_sol == x.get().round()).all()
    model.do_math().to_lp('lp_test')
    gmodel = gp.read('lp_test.lp')
    gmodel.optimize()
    assert model.solution.objval == gmodel.objVal
    assert model.optimal()

    model.solve(msk)
    assert (x_sol == x.get().round()).all()
    assert model.optimal()

    # model.solve(cpx)
    # assert (x_sol == x.get().round()).all()
    # assert model.optimal()

    model.solve(cpt)
    assert (x_sol == x.get().round()).all()
    assert model.optimal()


def test_socp():

    n = 10
    c = rd.randn(n)

    model = socp.Model()
    x = model.dvar(n)
    model.max(c@x)
    model.st(rso.norm(x) <= 1)

    x_sol = c / (c**2).sum()**0.5
    objval = c @ x_sol

    with pytest.warns(UserWarning):
        model.solve()
    assert not model.optimal()

    """
    with pytest.warns(UserWarning):
        model.solve(clp)
    assert not model.optimal()
    """

    with pytest.warns(UserWarning):
        model.solve(ort)
    assert not model.optimal()

    model.solve(eco)
    assert abs(model.get() - objval) < 1e-6
    assert (abs(x_sol - x.get()) < 1e-3).all()
    assert model.optimal()

    model.solve(grb)
    assert abs(model.get() - objval) < 1e-6
    assert (abs(x_sol - x.get()) < 1e-3).all()
    assert model.optimal()
    model.do_math().to_lp('lp_test')
    gmodel = gp.read('lp_test.lp')
    gmodel.optimize()
    assert model.solution.objval == gmodel.objVal

    model.solve(msk)
    assert model.optimal()
    assert abs(model.get() - objval) < 1e-6
    assert (abs(x_sol - x.get()) < 1e-3).all()

    # model.solve(cpx)
    # assert model.optimal()
    # assert abs(model.get() - objval) < 1e-6
    # assert (abs(x_sol - x.get()) < 1e-3).all()

    model.solve(cpt)
    assert model.optimal()
    assert abs(model.get() - objval) < 1e-6
    assert (abs(x_sol - x.get()) < 1e-3).all()


def test_mip_socp():

    n = 5
    rd.seed(5)
    c = rd.randn(n)

    model = ro.Model()
    a = model.dvar(vtype='i')
    x = model.dvar(n)
    model.max(0.001*a + c@x)
    model.st(a <= 10*c@x)
    model.st(rso.norm(x) <= 1)
    model.st(x <= 100, x >= -100)

    x_sol = c / (c**2).sum()**0.5
    objval = 0.001*int(10*c@x_sol) + c@x_sol

    with pytest.warns(UserWarning):
        model.solve()

    """
    with pytest.warns(UserWarning):
        model.solve(clp)
    """

    with pytest.warns(UserWarning):
        model.solve(ort)

    model.solve(eco)
    assert abs(model.get() - objval) < 1e-3
    assert (abs(x_sol - x.get()) < 1e-3).all()

    model.solve(grb)
    assert abs(model.get() - objval) < 1e-3
    assert (abs(x_sol - x.get()) < 1e-3).all()
    model.do_math().to_lp('lp_test')
    gmodel = gp.read('lp_test.lp')
    gmodel.optimize()
    assert model.solution.objval == gmodel.objVal

    model.solve(msk)
    assert abs(model.get() - objval) < 1e-3
    assert (abs(x_sol - x.get()) < 1e-3).all()

    # model.solve(cpx)
    # assert abs(model.get() - objval) < 1e-3
    # assert (abs(x_sol - x.get()) < 1e-3).all()

    model.solve(cpt)
    assert abs(model.get() - objval) < 1e-3
    assert (abs(x_sol - x.get()) < 1e-3).all()


def test_no_solution():

    model = ro.Model()
    x = model.dvar(5)
    y = model.dvar(5)

    model.min(x.sum())
    model.st(x == 1.5*np.ones(5))
    model.st(y <= 0)
    model.st(x <= y)

    with pytest.warns(UserWarning):
        model.solve()

    """
    with pytest.warns(UserWarning):
        model.solve(clp)
    """

    with pytest.warns(UserWarning):
        model.solve(ort)

    with pytest.warns(UserWarning):
        model.solve(eco)

    with pytest.warns(UserWarning):
        model.solve(grb)

    # with pytest.warns(UserWarning):
    #     model.solve(cpx)

    with pytest.warns(UserWarning):
        model.solve(msk)

    with pytest.warns(UserWarning):
        model.solve(cpt)

    assert not model.optimal()

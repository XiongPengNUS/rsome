"""
This module is used as an interface to call the Gurobi solver for solving
(mixed-integer) linear or second-order cone programs.
"""

from gurobipy import GurobiError
import gurobipy as gp
import numpy as np
import warnings
import time
from .socp import SOCProg
from .lp import Solution


version = '.'.join([str(d) for d in gp.gurobi.version()])
name = 'Gurobi'
info = f'{name} {version}'


def solve(formula, display=True, log=False, params={}):

    try:
        if formula.xmat:
            warnings.warn('The SOCP solver ignores exponential cone constraints. ')
        if formula.lmi:
            warnings.warn('The SOCP solver ignores semidefinite cone constraints. ')
    except AttributeError:
        pass

    nv = formula.linear.shape[1]
    vtype = list(formula.vtype)

    grb = gp.Model()
    x = grb.addMVar(nv, lb=formula.lb, ub=formula.ub, vtype=vtype)

    indices_eq = (formula.sense == 1)
    indices_ineq = (formula.sense == 0)
    linear_eq = formula.linear[indices_eq, :]
    linear_ineq = formula.linear[indices_ineq, :]
    const_eq = formula.const[indices_eq]
    const_ineq = formula.const[indices_ineq]
    if len(indices_eq) > 0:
        c_eq = grb.addMConstr(linear_eq, x, '=', const_eq)
        # grb.addMConstrs(linear_eq, x, '=', const_eq)
    if len(indices_ineq) > 0:
        c_ineq = grb.addMConstr(linear_ineq, x, '<', const_ineq)
        # grb.addMConstrs(linear_ineq, x, '<', const_ineq)

    if isinstance(formula, SOCProg):
        for constr in formula.qmat:
            index_right = constr[0:1]
            index_left = constr[1:]
            A = np.eye(len(index_left))
            grb.addConstr(x[index_left] @ A @ x[index_left] <=
                          x[index_right] @ x[index_right])

    grb.setObjective(formula.obj @ x)

    if not log:
        grb.setParam('LogToConsole', 0)
    try:
        for param, value in params.items():
            if eval('grb.Params.{}'.format(param)) is None:
                raise ValueError('Unknown parameter')
            grb.setParam(param, value)

    except (TypeError, ValueError):
        raise ValueError('Incorrect parameters or values.')
    if display:
        print('Being solved by Gurobi...', flush=True)
        time.sleep(0.2)
    grb.optimize()
    if display:
        print('Solution status: {0}'.format(grb.Status))
        print('Running time: {0:0.4f}s'.format(grb.Runtime))

    try:
        pi = np.ones(formula.linear.shape[0]) * np.nan
        upi = np.zeros(nv)
        lpi = np.zeros(nv)
        pi[indices_eq] = c_eq.pi
        pi[indices_ineq] = c_ineq.pi
        rc = x.rc
        upi[rc < 0] = rc[rc < 0]
        lpi[rc > 0] = rc[rc > 0]
        y = {'pi': pi, 'upi': upi, 'lpi': lpi}
    except GurobiError:
        y = None

    try:
        solution = Solution('Gurobi', grb.ObjVal, np.array(grb.getAttr('X')),
                            grb.Status, grb.Runtime, y=y)
    except AttributeError:
        warnings.warn('Fail to find the optimal solution.')
        # solution = None
        solution = Solution('Gurobi', np.nan, None, grb.Status, grb.Runtime)

    return solution

"""
This module is used as an interface to call the CVXPY solver for solving
(mixed-integer) linear or second-order cone programs.
"""

import cvxpy as cp
import numpy as np
import warnings
import time
from .socp import SOCProg
from .lp import Solution


def solve(formula, display=True, export=False, params={}):

    num_col = formula.linear.shape[1]
    indices_int,  = np.where(np.isin(formula.vtype, 'BI'))
    integer = [(index, ) for index in indices_int]
    x = cp.Variable(shape=num_col, integer=integer)

    indices_eq = formula.sense == 1
    indices_ineq = formula.sense == 0

    linear_eq = formula.linear[indices_eq, :]
    linear_ineq = formula.linear[indices_ineq, :]

    const_eq = formula.const[indices_eq]
    const_ineq = formula.const[indices_ineq]

    constraints = []
    if len(const_eq):
        constraints.append(linear_eq @ x == const_eq)
    if len(const_ineq):
        constraints.append(linear_ineq @ x <= const_ineq)

    indices_lb = formula.lb > - np.inf
    indices_ub = formula.ub < np.inf
    bounds = []
    if any(indices_lb):
        bounds.append(x[indices_lb] >= formula.lb[indices_lb])
    if any(indices_ub):
        bounds.append(x[indices_ub] <= formula.ub[indices_ub])

    indices_bin = formula.vtype == 'B'
    if any(indices_bin):
        bounds.append(x[indices_bin] >= 0)
        bounds.append(x[indices_bin] <= 1)

    if isinstance(formula, SOCProg):
        socs = [cp.SOC(x[ind[0]], x[ind[1:]])
                for ind in formula.qmat]
    else:
        socs = []

    prob = cp.Problem(cp.Minimize(formula.obj@x), constraints+bounds+socs)

    if display:
        print('Being solved by CVXPY...', flush=True)
        time.sleep(0.2)

    t0 = time.time()
    prob.solve(solver='ECOS_BB')
    stime = time.time() - t0

    if display:
        print('Solution status: {0}'.format(prob.status))
        print('Running time: {0:0.4f}s'.format(stime))

    try:
        solution = Solution(prob.value, x.value, prob.status)
    except AttributeError:
        warnings.warn('No feasible solution can be found.')
        solution = None

    return solution

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


def solve(formula, display=True, params={}):

    warnings.warn('This interface will be deprecated in future versions.',
                  FutureWarning)

    num_col = formula.linear.shape[1]
    indices_int, = np.where(formula.vtype == 'I')
    indices_bin, = np.where(formula.vtype == 'B')
    indices_con, = np.where(formula.vtype == 'C')
    x_vec = 0
    if len(indices_con) > 0:
        x = cp.Variable(shape=len(indices_con), integer=False)
        coeff_con = np.zeros((num_col, len(indices_con)))
        coeff_con[indices_con, range(len(indices_con))] = 1
        x_vec += coeff_con@x
    if len(indices_int) > 0:
        xin = cp.Variable(shape=len(indices_int), integer=True)
        coeff_int = np.zeros((num_col, len(indices_int)))
        coeff_int[indices_int, range(len(indices_int))] = 1
        x_vec += coeff_int@xin
    if len(indices_bin) > 0:
        xbin = cp.Variable(shape=len(indices_bin), boolean=True)
        coeff_bin = np.zeros((num_col, len(indices_bin)))
        coeff_bin[indices_bin, range(len(indices_bin))] = 1
        x_vec += coeff_bin@xbin

    indices_eq = formula.sense == 1
    indices_ineq = formula.sense == 0

    linear_eq = formula.linear[indices_eq, :]
    linear_ineq = formula.linear[indices_ineq, :]

    const_eq = formula.const[indices_eq]
    const_ineq = formula.const[indices_ineq]

    constraints = []
    if len(const_eq):
        constraints.append(linear_eq @ x_vec == const_eq)
    if len(const_ineq):
        constraints.append(linear_ineq @ x_vec <= const_ineq)

    indices_lb = formula.lb > - np.inf
    indices_ub = formula.ub < np.inf
    bounds = []
    for index in np.where(indices_lb)[0]:
        if index in indices_con:
            bounds.append(x[list(indices_con).index(index)] >= formula.lb[index])
        elif index in indices_int:
            bounds.append(xin[list(indices_int).index(index)] >= formula.lb[index])
        else:
            bounds.append(xbin[list(indices_bin).index(index)] >= formula.lb[index])
    for index in np.where(indices_ub)[0]:
        if index in indices_con:
            bounds.append(x[list(indices_con).index(index)] <= formula.ub[index])
        elif index in indices_int:
            bounds.append(xin[list(indices_int).index(index)] <= formula.ub[index])
        else:
            bounds.append(xbin[list(indices_bin).index(index)] <= formula.ub[index])

    if isinstance(formula, SOCProg):
        socs = [cp.SOC(x_vec[ind[0]], x_vec[ind[1:]])
                for ind in formula.qmat]
    else:
        socs = []

    prob = cp.Problem(cp.Minimize(formula.obj@x_vec), constraints+bounds+socs)

    if display:
        print('Being solved by CVXPY...', flush=True)
        time.sleep(0.2)

    t0 = time.time()
    status = prob.solve(solver='ECOS_BB')
    stime = time.time() - t0

    if display:
        print('Solution status: {0}'.format(prob.status))
        print('Running time: {0:0.4f}s'.format(stime))

    if status < np.inf:
        x_vec = np.zeros(num_col)
        if len(indices_con) > 0:
            x_vec[indices_con] = x.value
        if len(indices_int) > 0:
            x_vec[indices_int] = xin.value
        if len(indices_bin) > 0:
            x_vec[indices_bin] = xbin.value
        solution = Solution(prob.value, x_vec, prob.status, stime)
    else:
        warnings.warn('Fail to find the optimal solution.')
        solution = None

    return solution

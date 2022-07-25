"""
This module is used as an interface to call the COPT solver for solving
(mixed-integer) linear or second-order cone programs.
"""

import coptpy as cp
import numpy as np
import warnings
import time
from scipy.sparse import csc_matrix
from .socp import SOCProg
from .lp import Solution


def solve(formula, display=True, params={}):

    try:
        if formula.xmat:
            warnings.warn('The SOCP solver ignores exponential cone constraints. ')
    except AttributeError:
        pass

    env = cp.Envr()
    m = env.createModel()
    m.setParam(cp.COPT.Param.Logging, False)
    m.setParam(cp.COPT.Param.LogToConsole, False)
    

    c = formula.obj[0]
    A = formula.linear
    vtype = formula.vtype
    lhs = np.array([-cp.COPT.INFINITY]*formula.linear.shape[0])
    index_eq = formula.sense == 1
    lhs[index_eq] = formula.const[index_eq]
    rhs = formula.const

    lb = formula.lb
    ub = formula.ub
    lb[vtype=='B'] = 0
    ub[vtype=='B'] = 1

    m.loadMatrix(c, csc_matrix(A), lhs, rhs, lb, ub, vtype)

    if isinstance(formula, SOCProg):
        sc_dim = []
        sc_indices = []    
        for q in formula.qmat:
            sc_dim.append(len(q))
            sc_indices.extend(q)
        
        ncone = len(formula.qmat)
        if ncone > 0:
            m.loadCone(ncone, None, sc_dim, sc_indices)

    if display:
        print('Being solved by COPT...', flush=True)
        time.sleep(0.2)
    m.solve()
    if display:
        stime = m.getAttr(cp.COPT.attr.SolvingTime)
        if all(vtype == 'C'):
            status = m.getAttr(cp.COPT.attr.LpStatus)
        else:
            status = m.getAttr(cp.COPT.attr.MipStatus)
        print('Solution status: {0}'.format(status))
        print('Running time: {0:0.4f}s'.format(stime))

    try:
        x_sol = np.array(m.getValues())
        solution = Solution(x_sol[0], x_sol, status, stime)
    except cp.CoptError:
        warnings.warn('Fail to find the optimal solution.')
        solution = None

    return solution

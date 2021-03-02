from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import numpy as np
import warnings
import time
from .lp import Solution


def solve(formula, display=True, export=False, params={}):

    try:
        if formula.qmat:
            warnings.warn('SOC constriants are ignored in the LP solver. ')
    except AttributeError:
        pass

    obj = formula.obj.flatten()
    linear = formula.linear
    sense = formula.sense
    const = formula.const
    ub = formula.ub
    lb = formula.lb
    vtype = formula.vtype

    eq_linear = linear[sense == 1, :]
    eq_const = const[sense == 1]
    ineq_linear = linear[sense == 0, :]
    ineq_const = const[sense == 0]

    is_con = vtype == 'C'
    is_int = vtype == 'I'
    is_bin = vtype == 'B'

    s = CyClpSimplex()

    obj_expr = 0
    ncv = sum(is_con)
    if ncv:
        cv = s.addVariable('cv', ncv)
        s += cv <= ub[is_con]
        s += cv >= lb[is_con]
        obj_expr += CyLPArray(obj[is_con]) * cv

    niv = sum(is_int)
    if niv:
        iv = s.addVariable('iv', niv, isInt=True)
        s += iv <= ub[is_int]
        s += iv >= lb[is_int]
        obj_expr += CyLPArray(obj[is_int]) * iv

    nbv = sum(is_bin)
    if nbv:
        bv = s.addVariable('bv', nbv)
        s += bv <= 1
        s += bv >= 0
        obj_expr += CyLPArray(obj[is_bin]) * bv
    s.objective = obj_expr

    if eq_linear.shape[0] > 0:
        left = 0
        if ncv:
            left += eq_linear[:, is_con] * cv
        if niv:
            left += eq_linear[:, is_int] * iv
        if nbv:
            left += eq_linear[:, is_bin] * bv

        s += left == eq_const

    if ineq_linear.shape[0] > 0:
        left = 0
        if ncv:
            left += ineq_linear[:, is_con] * cv
        if niv:
            left += ineq_linear[:, is_int] * iv
        if nbv:
            left += ineq_linear[:, is_bin] * bv

        s += left <= ineq_const

    if display:
        print('Being solved by CyLP...', flush=True)
        time.sleep(0.2)
    t0 = time.time()
    status = s.primal()
    stime = time.time() - t0
    if display:
        print('Solution status: {0}'.format(status))
        print('Running time: {0:0.4f}s'.format(stime))

    try:
        x_sol = np.zeros(linear.shape[1])
        if ncv:
            x_sol[is_con] = s.primalVariableSolution['cv']
        if niv:
            x_sol[is_int] = s.primalVariableSolution['iv'].round()
        if nbv:
            x_sol[is_bin] = s.primalVariableSolution['bv'].round()

        solution = Solution(s.objectiveValue, x_sol, status)
    except AttributeError:
        warnings.warn('No feasible solution can be found.')
        solution = None

    return solution

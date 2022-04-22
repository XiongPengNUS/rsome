"""
This module is used as an interface to call the CPLEX solver for solving
(mixed-integer) linear or second-order cone programs.
"""

import cplex
import numpy as np
import warnings
import time
from .lp import Solution
from .socp import SOCProg


def solve(formula, display=True, params={}):

    try:
        if formula.xmat:
            warnings.warn('The SOCP solver ignores exponential cone constraints. ')
    except AttributeError:
        pass

    cpx = cplex.Cplex()

    # obj = formula.obj.flatten()
    linear = formula.linear
    row = linear.shape[0]
    spmat = [[linear.indices[linear.indptr[i]:linear.indptr[i + 1]].tolist(),
              linear.data[linear.indptr[i]:linear.indptr[i + 1]].tolist()]
             for i in range(row)]
    sense = ['E' if s == 1 else 'L' for s in formula.sense]
    vtype = [cpx.variables.type.integer if vt == 'I' else
             cpx.variables.type.binary if vt == 'B' else
             cpx.variables.type.continuous for vt in formula.vtype]

    if all(np.array(vtype) == cpx.variables.type.continuous):
        cpx.variables.add(obj=formula.obj.flatten(),
                          lb=formula.lb, ub=formula.ub)
    else:
        cpx.variables.add(obj=formula.obj.flatten(),
                          lb=formula.lb, ub=formula.ub, types=vtype)
    cpx.linear_constraints.add(lin_expr=spmat,
                               senses=sense, rhs=formula.const)

    if isinstance(formula, SOCProg):
        for cone in formula.qmat:
            cone_data = [-1] + [1] * (len(cone) - 1)
            cone = [int(index) for index in cone]
            q = cplex.SparseTriple(ind1=cone, ind2=cone, val=cone_data)
            cpx.quadratic_constraints.add(quad_expr=q)

    if display:
        print('Being solved by CPLEX...', flush=True)
        time.sleep(0.2)

    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    try:
        for param, value in params.items():
            text = 'cpx.parameters.' + param + '.set({0})'.format(value)
            eval(text)
    except (TypeError, ValueError, AttributeError):
        raise ValueError('Incorrect parameters or values.')

    t0 = time.time()
    cpx.solve()
    stime = time.time() - t0
    status = cpx.solution.get_status()
    if display:
        print('Solution status: {0}'.format(status))
        print('Running time: {0:0.4f}s'.format(stime))

    if status in [1, 6, 10, 11, 12, 13,
                  21, 22,
                  101, 102, 105, 107, 109, 111, 113, 116]:
        obj_val = cpx.solution.get_objective_value()
        x_sol = np.array(cpx.solution.get_values())
        solution = Solution(obj_val, x_sol, status, stime)
    else:
        warnings.warn('Fail to find the optimal solution.')
        solution = None

    return solution

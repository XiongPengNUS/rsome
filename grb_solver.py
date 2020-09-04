import gurobipy as gp
import numpy as np
import warnings
from .socp import SOCProg
from .lp import Solution


def solve(formula, display=True, export=False):

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
        grb.addMConstrs(linear_eq, x, '=', const_eq)
    if len(indices_ineq) > 0:
        grb.addMConstrs(linear_ineq, x, '<', const_ineq)

    if isinstance(formula, SOCProg):
        for constr in formula.qmat:
            index_right = constr[0]
            index_left = constr[1:]
            A = np.eye(len(index_left))
            grb.addConstr(x[index_left] @ A @ x[index_left] <=
                          x[index_right] @ x[index_right])

    grb.setObjective(formula.obj @ x)

    grb.setParam('LogToConsole', 0)
    if display:
        print('Being solved by Gurobi...')
    grb.optimize()
    if display:
        print('Solution status: {0}'.format(grb.Status))
        print('Running time: {0:0.4f}s'.format(grb.Runtime))

    if export:
        grb.write("out.lp")

    try:
        solution = Solution(grb.ObjVal, grb.getAttr('X'), grb.Status)
    except:
        warnings.warn('No feasible solution can be found.')
        solution = None

    return solution

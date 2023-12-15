"""
This module is used as an interface to call the MOSEK solver for solving (mixed-
integer) linear or conic programs.
"""

from mosek.fusion import *
import numpy as np
from scipy.sparse import coo_matrix
from .socp import SOCProg
from .gcp import GCProg
from .lp import Solution
import warnings
import time
import sys


version = Model.getVersion()
name = 'Mosek'
info = f'{name} {version}'


def solve(form, display=True, log=False, params={}):

    if isinstance(form, (SOCProg, GCProg)):
        qmat = form.qmat
    else:
        qmat = []
    if isinstance(form, GCProg):
        xmat = form.xmat
        lmi = form.lmi
    else:
        xmat = []
        lmi = []

    idx_cont = [i for i, v in enumerate(form.vtype) if v == 'C']
    idx_bin = [i for i, v in enumerate(form.vtype) if v == 'B']
    idx_int = [i for i, v in enumerate(form.vtype) if v == 'I']

    idx_ub = [i for i, v in enumerate(form.ub) if v != np.inf]
    idx_lb = [i for i, v in enumerate(form.lb) if v != -np.inf]

    is_eq = (form.sense == 1)
    coo_linear_ineq = coo_matrix(form.linear[~is_eq])
    linear_ineq = Matrix.sparse(coo_linear_ineq.shape[0], coo_linear_ineq.shape[1],
                                coo_linear_ineq.row, coo_linear_ineq.col,
                                coo_linear_ineq.data)
    coo_linear_eq = coo_matrix(form.linear[is_eq])
    linear_eq = Matrix.sparse(coo_linear_eq.shape[0], coo_linear_eq.shape[1],
                              coo_linear_eq.row, coo_linear_eq.col, coo_linear_eq.data)

    const_ineq = form.const[~is_eq]
    const_eq = form.const[is_eq]

    num_constr, num_var = form.linear.shape
    with Model() as M:
        num_cont = len(idx_cont)
        xc = M.variable("xc", num_cont)
        x = Expr.mul(Matrix.sparse(num_var, num_cont, idx_cont,
                                   list(range(num_cont)), np.ones(num_cont)), xc)
        if idx_bin:
            num_bin = len(idx_bin)
            xb = M.variable("xb", num_bin, Domain.binary())
            x = Expr.add(Expr.mul(Matrix.sparse(num_var, num_bin, idx_bin,
                                                list(range(num_bin)),
                                                np.ones(num_bin)), xb), x)

        if idx_int:
            num_int = len(idx_int)
            xi = M.variable("xi", num_int, Domain.integral(Domain.unbounded()))
            x = Expr.add(Expr.mul(Matrix.sparse(num_var, num_int, idx_int,
                                                list(range(num_int)),
                                                np.ones(num_int)), xi), x)

        obj_expr = Expr.mul(form.obj.reshape((1, form.obj.size)), x)
        M.objective(ObjectiveSense.Minimize, obj_expr)
        c_ineq = M.constraint(Expr.mul(linear_ineq, x), Domain.lessThan(const_ineq))
        c_eq = M.constraint(Expr.mul(linear_eq, x), Domain.equalsTo(const_eq))
        c_ub = M.constraint(x.pick(idx_ub), Domain.lessThan(form.ub[idx_ub]))
        c_lb = M.constraint(x.pick(idx_lb), Domain.greaterThan(form.lb[idx_lb]))

        for q in qmat:
            M.constraint(x.pick(q), Domain.inQCone())

        for e in xmat:
            M.constraint(x.pick([e[1], e[2], e[0]]), Domain.inPExpCone())

        for p in lmi:
            temp_coo = coo_matrix(p['linear'])
            psd_linear = Matrix.sparse(temp_coo.shape[0], coo_linear_ineq.shape[1],
                                       temp_coo.row, temp_coo.col, temp_coo.data)
            psd_const = p['const'].flatten()
            dim = p['dim']

            left = Expr.reshape(Expr.sub(Expr.mul(psd_linear, x), psd_const), dim, dim)
            M.constraint(left, Domain.inPSDCone(dim))

        for param, value in params.items():
            M.setSolverParam(param, value)

        if log:
            M.setLogHandler(sys.stdout)

        if display:
            print('Being solved by Mosek...', flush=True)
            time.sleep(0.2)
        t0 = time.time()
        M.solve()
        stime = time.time() - t0
        stats_string = M.getPrimalSolutionStatus().__str__()
        status = stats_string.split('.')[1]
        if display:
            print('Solution status: {0}'.format(status))
            print('Running time: {0:0.4f}s'.format(stime))

        if status == 'Optimal':
            x_sol = coo_matrix((np.ones(num_cont), (idx_cont, np.arange(num_cont))),
                               (num_var, num_cont)) @ xc.level()

            if idx_bin:
                x_sol += coo_matrix((np.ones(num_bin), (idx_bin, np.arange(num_bin))),
                                    (num_var, num_bin)) @ xb.level()
            if idx_int:
                x_sol += coo_matrix((np.ones(num_int), (idx_int, np.arange(num_int))),
                                    (num_var, num_int)) @ xi.level()

            if all(form.vtype == 'C'):
                pi = np.ones(num_constr) * np.nan
                upi = np.zeros(num_var)
                lpi = np.zeros(num_var)
                if not idx_bin and not idx_int:
                    pi[is_eq] = c_eq.dual()
                    pi[~is_eq] = c_ineq.dual()

                    upi[idx_ub] = c_ub.dual()
                    lpi[idx_lb] = c_lb.dual()

                y = {'pi': pi, 'upi': upi, 'lpi': lpi}
            else:
                y = None
            solution = Solution('Mosek', x_sol @ form.obj, x_sol, status, stime, y=y)
        else:
            warnings.warn('Fail to find the optimal solution.')
            # solution = None
            solution = Solution('Mosek', np.nan, None, status, stime)

    return solution

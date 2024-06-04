"""
This module is used as an interface to call the COPT solver for solving
(mixed-integer) linear or second-order cone programs.
"""

import coptpy as cp
import numpy as np
import warnings
import time
from scipy.sparse import csr_matrix
from .socp import SOCProg
from .gcp import GCProg
from .lp import Solution


ds = [cp.GetCoptVersion(i) for i in range(5) if cp.GetCoptVersion(i) >= 0]
version = '.'.join([str(d) for d in ds])
name = 'COPT'
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

    try:
        if xmat:
            warnings.warn('The conic solver ignores exponential cone constraints. ')
    except AttributeError:
        pass

    idx_cont = [i for i, v in enumerate(form.vtype) if v == 'C']
    idx_bin = [i for i, v in enumerate(form.vtype) if v == 'B']
    idx_int = [i for i, v in enumerate(form.vtype) if v == 'I']

    idx_ub = [i for i, v in enumerate(form.ub) if v != np.inf]
    idx_lb = [i for i, v in enumerate(form.lb) if v != -np.inf]

    is_eq = (form.sense == 1)

    linear_ineq = form.linear[~is_eq]
    linear_eq = form.linear[is_eq]
    num_ineq = linear_ineq.shape[0]

    const_ineq = form.const[~is_eq]
    const_eq = form.const[is_eq]

    num_constr, num_var = form.linear.shape

    envconfig = cp.EnvrConfig()
    envconfig.set('nobanner', '1')
    env = cp.Envr(envconfig)
    m = env.createModel()
    m.setParam(cp.COPT.Param.Logging, log)
    m.setParam(cp.COPT.Param.LogToConsole, False)

    num_cont = len(idx_cont)
    xc = m.addMVar(num_cont, lb=-1e30)

    mat_cont = csr_matrix((np.ones(num_cont), (idx_cont, range(num_cont))),
                          (num_var, num_cont))
    xx = mat_cont@xc

    if idx_bin:
        num_bin = len(idx_bin)
        xb = m.addMVar(num_bin, vtype='B')
        mat_bin = csr_matrix((np.ones(num_bin), (idx_bin, range(num_bin))),
                             (num_var, num_bin))
        xx += mat_bin@xb

    if idx_int:
        num_int = len(idx_int)
        xi = m.addMVar(num_int, vtype='I', lb=-1e30)
        mat_int = csr_matrix((np.ones(num_int), (idx_int, range(num_int))),
                             (num_var, num_int))
        xx += mat_int@xi

    m.setObjective(form.obj@xx, sense=cp.COPT.MINIMIZE)
    if linear_ineq.shape[0] > 0:
        m.addConstrs(linear_ineq@xx - const_ineq <= 0)
    if linear_eq.shape[0] > 0:
        m.addConstrs(linear_eq@xx - const_eq == 0)

    if idx_ub:
        m.addConstrs(xx[idx_ub] <= form.ub[idx_ub])
    if idx_lb:
        m.addConstrs(xx[idx_lb] >= form.lb[idx_lb])

    for idx_q in qmat:
        var_list = []
        for idx in idx_q:
            if form.vtype[idx] == 'C':
                var_list.extend(xc[idx_cont.index(idx)].tolist())
            elif form.vtype[idx] == 'B':
                var_list.extend(xb[idx_bin.index(idx)].tolist())
            elif form.vtype[idx] == 'I':
                var_list.extend(xi[idx_int.index(idx)].tolist())
        m.addCone(var_list, cp.COPT.CONE_QUAD)

    for sdc in lmi:
        dim = sdc['dim']
        Xbar = m.addPsdVars(dim)
        num_col = sdc['linear'].shape[1]
        lhs = (sdc['linear']@xx[:num_col]).reshape((dim, dim)) + sdc['const']
        for i in range(dim):
            for j in range(i+1):
                matmul = m.addSparseMat(dim, [(i, j, 1)])
                if i == j:
                    m.addConstr(matmul*Xbar - lhs[i].tolist()[j] == 0)
                else:
                    m.addConstr(matmul*Xbar - 2*lhs[i].tolist()[j] == 0)

    if display:
        print('', 'Being solved by COPT...', sep='', flush=True)
        time.sleep(0.2)
    m.solve()

    stime = m.getAttr(cp.COPT.attr.SolvingTime)
    status = m.status
    if display:
        print('Solution status: {0}'.format(status))
        print('Running time: {0:0.4f}s'.format(stime))

    try:
        x_sol = np.array(m.getValues())
        objval = form.obj @ x_sol

        if all(form.vtype == 'C'):
            res = m.getLpSolution()
            pi = np.ones(num_constr) * np.nan
            pi[~is_eq] = res[2][:num_ineq]
            pi[is_eq] = res[2][num_ineq:num_constr]
            upi = np.zeros(num_var)
            lpi = np.zeros(num_var)
            upi[idx_ub] = res[2][num_constr:num_constr+len(idx_ub)]
            lpi[idx_lb] = res[2][num_constr+len(idx_ub):]

            y = {'pi': pi, 'upi': upi, 'lpi': lpi}
        else:
            y = None

        xx_sol = np.zeros(len(form.vtype))
        xx_sol[idx_cont] = x_sol[:len(idx_cont)]
        if idx_bin:
            xx_sol[idx_bin] = x_sol[len(idx_cont): len(idx_cont)+len(idx_bin)]
        if idx_int:
            xx_sol[idx_int] = x_sol[len(idx_cont)+len(idx_bin):]

        solution = Solution('COPT', objval, xx_sol, status, stime, y=y)
    except cp.CoptError:
        warnings.warn('Fail to find the optimal solution.')
        # solution = None
        solution = Solution('COPT', np.nan, None, status, stime)

    return solution

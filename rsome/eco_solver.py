"""
This module is used as an interface to call the ECOS solver for solving
(mixed-integer) linear, second-order cone, or exponential cone programs.
"""

import ecos
import numpy as np
import warnings
import time
import scipy.sparse as sp
from .socp import SOCProg
from .gcp import GCProg
from .lp import Solution


def solve(formula, display=True, params={}):

    bool_idx = [i for i in range(len(formula.vtype)) if formula.vtype[i] == 'B']
    int_idx = [i for i in range(len(formula.vtype)) if formula.vtype[i] == 'I']

    cols = formula.linear.shape[1]
    eq_idx = np.argwhere(formula.sense == 1).flatten()
    ineq_idx = np.argwhere(formula.sense == 0).flatten()
    num_ineq = len(ineq_idx)

    c = formula.obj.flatten()

    Gl = formula.linear[ineq_idx]

    zlb_idx = np.argwhere(formula.lb > -np.inf).flatten()
    num_zlb = len(zlb_idx)
    Glb = sp.csr_matrix((-np.ones(num_zlb),
                         (np.arange(num_zlb, dtype='int'), zlb_idx)),
                        (num_zlb, cols))
    zub_idx = np.argwhere(formula.ub < np.inf).flatten()
    num_zub = len(zub_idx)
    Gub = sp.csr_matrix((np.ones(num_zub),
                         (np.arange(num_zub, dtype='int'), zub_idx)),
                        (num_zub, cols))

    Gsc = []
    sc_dim = []
    qmat = formula.qmat if isinstance(formula, SOCProg) else []
    for q in qmat:
        num = len(q)
        socone = sp.csc_matrix((-np.ones(num),
                                (np.arange(num, dtype='int'), q)),
                               (num, cols))
        Gsc.append(socone)
        sc_dim.append(num)

    Gec = []
    xmat = formula.xmat if isinstance(formula, GCProg) else []
    for e in xmat:
        expcone = sp.csc_matrix((-np.ones(3),
                                 (np.arange(3, dtype='int'), e)),
                                (3, cols))
        Gec.append(expcone)

    G = sp.csc_matrix(sp.vstack([Gl, Glb, Gub] + Gsc + Gec))
    h = np.hstack((formula.const[ineq_idx],
                   -formula.lb[zlb_idx],
                   formula.ub[zub_idx],
                   np.zeros(sum(sc_dim)), np.zeros(len(xmat)*3)))

    dims = {'l': num_ineq + num_zlb + num_zub,
            'q': sc_dim, 'e': len(xmat)}
    if len(eq_idx) > 0:
        A = sp.csc_matrix(formula.linear[eq_idx])
        b = formula.const[eq_idx]
    else:
        A = b = None

    if display:
        print('Being solved by ECOS...', flush=True)
        time.sleep(0.2)

    if len(bool_idx) + len(int_idx) == 0:
        sol = ecos.solve(c, G, h, dims, A, b)
    else:
        sol = ecos.solve(c, G, h, dims, A, b,
                         bool_vars_idx=bool_idx, int_vars_idx=int_idx,
                         mi_max_iters=100000000)
    info = sol['info']
    stime = info['timing']['runtime']
    status = info['infostring']

    if display:
        print('Solution status: {0}'.format(status))
        print('Running time: {0:0.4f}s'.format(stime))

    if info['exitFlag'] in [0, 10]:
        x_vec = sol['x']
        solution = Solution(info['pcost'], x_vec, status, stime)
    else:
        warnings.warn('Fail to find the optimal solution.')
        solution = None

    return solution

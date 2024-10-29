"""
This module is used as an interface to call the COPT solver for solving
(mixed-integer) linear or second-order cone programs.
"""

import coptpy as cp
import numpy as np
import warnings
import time
from scipy.sparse import csr_matrix, csc_matrix, vstack, hstack, identity
from .socp import SOCProg
from .gcp import GCProg
from .lp import Solution


ds = [cp.GetCoptVersion(i) for i in range(5) if cp.GetCoptVersion(i) >= 0]
version = '.'.join([str(d) for d in ds])
name = 'COPT'
info = f'{name} {version}'


def solve(form, display=True, log=False, params={}):

    envconfig = cp.EnvrConfig()
    envconfig.set('nobanner', '1')
    env = cp.Envr(envconfig)
    m = env.createModel()
    if not log:
        m.setParam(cp.COPT.Param.Logging, log)
        m.setParam(cp.COPT.Param.LogToConsole, False)

    if isinstance(form, GCProg):
        lmi = form.lmi
        is_sdp = len(lmi) > 0
    else:
        is_sdp = False
    if is_sdp:
        vtype = form.vtype
        if (vtype != 'C').any():
            raise ValueError('The solver does not support mixed-integer SDP.')
        c = form.obj
        lb = form.lb
        ub = form.ub
        idx_lb = np.argwhere(lb > -np.inf).flatten().tolist()
        idx_ub = np.argwhere(ub < np.inf).flatten().tolist()
        num_lb = len(idx_lb)
        num_ub = len(idx_ub)

        A = form.linear
        new_cols = A.shape[1] + sum([mc['dim']*(mc['dim'] + 1)//2 for mc in lmi])
        cn = np.concatenate((c, np.zeros(new_cols - len(c))))

        b = form.const
        sense = form.sense
        Aeq = A[sense == 1, :]
        Aeq.resize((Aeq.shape[0], new_cols))
        Aineq = A[sense == 0, :]
        Aineq.resize((Aineq.shape[0], new_cols))
        Abound = csr_matrix(([-1]*num_lb + [1]*num_ub,
                             (np.arange(num_lb + num_ub), idx_lb + idx_ub)),
                            shape=[num_lb + num_ub, new_cols])

        qmat = form.qmat
        Asoc = []
        q_dims = []
        for q in qmat:
            Asoc.append(csr_matrix(([-1] * len(q), (np.arange(len(q)), q)),
                                   shape=(len(q), new_cols)))
            q_dims.append(len(q))

        xmat = form.xmat
        Aexc = []
        Axbound = []
        for e in xmat:
            xvalues = [-1, 1, 1, 1]
            ix, jx = [0, 1, 1, 2], [e[1], e[0], e[2], e[2]]
            Aexc.append(csr_matrix((xvalues, (ix, jx)), shape=(3, new_cols)))
            Axbound.append(csr_matrix(([-1], ([0], [e[2]])),
                                      shape=(1, new_cols)))
        ep_num = len(xmat)

        Aeqmats = []
        beqvecs = []
        Pmats = []
        s_dims = []
        col_count = A.shape[1]
        row_count = 0
        for mc in lmi:
            dim = mc['dim']
            vdim = dim * (1+dim) // 2
            i, j = np.triu_indices(dim)
            k = i*dim + j
            Aeq_psd = mc['linear'][k, :]
            Aeq_psd.resize((vdim, col_count))
            right_mat = - csr_matrix(identity(vdim))
            Aeq_psd = hstack((Aeq_psd, right_mat))
            Aeq_psd.resize((Aeq_psd.shape[0], new_cols))
            Aeqmats.append(Aeq_psd)
            beqvecs.extend((mc['const'][i, j]).tolist())
            values = - np.ones(vdim)
            k = np.arange(vdim)
            Pmats.append(csr_matrix((values, (k, col_count + k)),
                                    shape=(vdim, new_cols)))
            col_count += vdim
            row_count += vdim
            s_dims.append(dim)

        Pvecs = [0] * col_count if lmi else []
        Amat = vstack([Aeq] + Aeqmats + [Aineq, Abound] + Axbound +
                      Asoc + Aexc + Pmats).transpose()
        bvec = np.array(b[sense == 1].tolist() + beqvecs +
                        b[sense == 0].tolist() +
                        (-lb[idx_lb]).tolist() + (ub[idx_ub]).tolist() +
                        [0]*ep_num + [0]*sum(q_dims) + [0]*(3*ep_num) + Pvecs)

        dims = {'f': int(sum(sense)) + row_count,
                'l': int(sum(1-sense) + num_lb + num_ub + ep_num),
                'q': q_dims,
                'ep': ep_num,
                's': s_dims,
                'p': []}
        m.loadConeMatrix(bvec, Amat, -cn, dims)

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
            x_sol = m.getDuals()
            objval = c @ x_sol

            solution = Solution('COPT', objval, x_sol, status, stime, y=None)
        except cp.CoptError:
            warnings.warn('Fail to find the optimal solution.')
            solution = Solution('COPT', np.nan, None, status, stime)
    else:
        c = form.obj
        A = form.linear
        vtype = form.vtype
        lhs = np.array([-cp.COPT.INFINITY]*A.shape[0])
        index_eq = form.sense == 1
        lhs[index_eq] = form.const[index_eq]
        rhs = form.const
        lb = form.lb
        ub = form.ub
        lb[lb == -np.inf] = -cp.COPT.INFINITY
        ub[ub == np.inf] = cp.COPT.INFINITY
        lb[vtype == 'B'] = 0
        ub[vtype == 'B'] = 1
        m.loadMatrix(c, csc_matrix(A), lhs, rhs, lb, ub, vtype)

        if isinstance(form, SOCProg):
            sc_dim = []
            sc_indices = []
            for q in form.qmat:
                sc_dim.append(len(q))
                sc_indices.extend(q)
            ncone = len(form.qmat)
            if ncone > 0:
                m.loadCone(ncone, None, sc_dim, sc_indices)

        if isinstance(form, GCProg):
            indices = []
            for e in form.xmat:
                indices.extend([e[1], e[2], e[0]])
            ncone = len(form.xmat)
            if ncone > 0:
                m.loadExpCone(ncone, [cp.COPT.EXPCONE_PRIMAL]*ncone, indices)

        if display:
            print('', 'Being solved by COPT...', sep='', flush=True)
            time.sleep(0.2)
        m.solve()
        stime = m.getAttr(cp.COPT.attr.SolvingTime)
        if all(vtype == 'C'):
            status = m.getAttr(cp.COPT.attr.LpStatus)
        else:
            status = m.getAttr(cp.COPT.attr.MipStatus)
        if display:
            print('Solution status: {0}'.format(status))
            print('Running time: {0:0.4f}s'.format(stime))

        try:
            x_sol = np.array(m.getValues())
            objval = c @ x_sol
            solution = Solution('COPT', objval, x_sol, status, stime, y=None)
        except cp.CoptError:
            warnings.warn('Fail to find the optimal solution.')
            solution = Solution('COPT', np.nan, None, status, stime)

    return solution

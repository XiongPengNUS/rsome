from .lp import CvxConstr, PCvxConstr, ExpConstr, KLConstr
from .socp import Model as SOCModel
from .socp import SOCProg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections.abc import Iterable
from .subroutines import flat, rso_broadcast


class Model(SOCModel):

    def __init__(self, nobj=False, mtype='R', name=None, top=None):

        super().__init__(nobj, mtype, name, top)
        self.exp_constr = []
        self.other_constr = []

    def reset(self):

        self.lin_constr = []
        self.pws_constr = []
        self.cone_constr = []
        self.exp_constr = []
        self.other_constr = []
        self.bounds = []
        self.aux_constr = []
        self.aux_bounds = []
        self.cvx_constr = []

    def st(self, constr):

        if isinstance(constr, Iterable):
            for item in constr:
                self.st(item)
        elif isinstance(constr, ExpConstr):
            self.exp_constr.append(constr)
        elif isinstance(constr, CvxConstr):
            if constr.xtype in 'XLP':
                self.other_constr.append(constr)
                # affine_out = constr.affine_out * (1/constr.multiplier)
                # exprs_list = rso_broadcast(constr.affine_in, affine_out)
                # for exprs in exprs_list:
                #     exp_cone_constr = ExpConstr(constr.model,
                #                                 exprs[1], 1, exprs[0])
                #     self.exp_constr.append(exp_cone_constr)
            else:
                super().st(constr)
        elif isinstance(constr, KLConstr):
            self.other_constr.append(constr)
        else:
            super().st(constr)

    def do_math(self, primal=True, refresh=True, obj=True):
        """
        Return the linear, second-order cone, or exponential
        cone programming problem as the standard formula of
        the model.

        Parameters
        ----------
        primal : bool, default True
            Specify whether return the primal formula of the model.
            If primal=False, the method returns the daul formula.

        refresh : bool
            Leave the argument unspecified.

        obj : bool
            Leave the argument unspecified.

        Returns
        -------
        prog : GCProg
            An exponential cone programming problem.
        """

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal

            if refresh:
                self.auxs = []
                self.aux_constr = []
                self.aux_bounds = []
                self.last = self.vars[-1].first + self.vars[-1].size

            more_exp = []
            if self.obj is not None:
                obj_constr = (self.vars[0] - self.sign * self.obj >= 0)
                if isinstance(obj_constr, CvxConstr):
                    constr = obj_constr
                    """
                    if constr.xtype == 'X':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exp_cone_constr = ExpConstr(constr.model, 
                                                    constr.affine_in, 
                                                    -constr.affine_out, 1)
                        more_exp.append(exp_cone_constr)
                    elif constr.xtype == 'L':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exp_cone_constr = ExpConstr(constr.model,
                                                    affine_out,
                                                    constr.affine_in, 1)
                        more_exp.append(exp_cone_constr)
                    """
                    if constr.xtype in 'XLP':
                        self.other_constr.append(constr)
            

            for constr in self.other_constr:
                if isinstance(constr, KLConstr):
                    ns = constr.p.size
                    aux_var = self.dvar(ns, aux=True)
                    self.aux_constr.append(aux_var.sum() <= constr.r)
                    for s in range(ns):
                        ps = constr.phat[s]
                        p = constr.p[s]

                        entro_constr = aux_var[s] * (1/ps) >= -(p * (1/ps)).entropy()
                        exp_constr = ExpConstr(entro_constr.model,
                                               entro_constr.affine_out,
                                               1, entro_constr.affine_in)

                        more_exp.append(exp_constr)
                elif isinstance(constr, PCvxConstr):
                    if constr.xtype == 'X':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exprs_list = rso_broadcast(constr.affine_in,
                                                   constr.affine_scale,
                                                   affine_out)
                        for exprs in exprs_list:
                            exp_cone_constr = ExpConstr(constr.model,
                                                        exprs[0], -exprs[2], exprs[1])
                            self.exp_constr.append(exp_cone_constr)
                    elif constr.xtype == 'L':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exprs_list = rso_broadcast(constr.affine_in,
                                                   constr.affine_scale,
                                                   affine_out)
                        for exprs in exprs_list:
                            exp_cone_constr = ExpConstr(constr.model, 
                                                        exprs[2], exprs[0], exprs[1])
                            self.exp_constr.append(exp_cone_constr)
                elif isinstance(constr, CvxConstr):
                    if constr.xtype == 'P':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        aux_var = self.dvar(constr.affine_in.shape)
                        self.aux_constr.append(aux_var.sum() >= affine_out)
                        ns = constr.affine_in.size
                        affine_in = constr.affine_in.reshape(ns)
                        affine_aux = aux_var.to_affine().reshape(ns)
                        for s in range(ns):
                            exp_cone_constr = ExpConstr(constr.model, 
                                                        affine_aux[s], 
                                                        1, affine_in[s])
                            more_exp.append(exp_cone_constr)
                    elif constr.xtype == 'X':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exprs_list = rso_broadcast(constr.affine_in, affine_out)
                        for exprs in exprs_list:
                            exp_cone_constr = ExpConstr(constr.model, 
                                                        exprs[0], -exprs[1], 1)
                            self.exp_constr.append(exp_cone_constr)
                    elif constr.xtype == 'L':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exprs_list = rso_broadcast(constr.affine_in, affine_out)
                        for exprs in exprs_list:
                            exp_cone_constr = ExpConstr(constr.model, 
                                                        exprs[1], exprs[0], 1)
                            self.exp_constr.append(exp_cone_constr)

            xmat = []
            for constr in self.exp_constr + more_exp:
                aux_var = self.dvar(3, aux=True)
                self.aux_constr.append(aux_var[0] - constr.expr1 == 0)
                self.aux_constr.append(aux_var[1] - constr.expr2 <= 0)
                self.aux_constr.append(aux_var[2] - constr.expr3 == 0)
                xmat.append(list(range(aux_var.first, aux_var.first + 3)))

            formula = super().do_math(primal=True, refresh=False, obj=obj)
            formula = GCProg(formula.linear, formula.const, formula.sense,
                             formula.vtype, formula.ub, formula.lb,
                             formula.qmat, xmat, formula.obj)
            self.primal = formula
            self.pupdate = False

            return formula

        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

            primal = self.do_math(obj=obj)

            dual_socp = super().do_math(primal=False, refresh=False, obj=obj)

            if len(primal.xmat) == 0:
                formula = GCProg(dual_socp.linear, dual_socp.const, dual_socp.sense,
                                 dual_socp.vtype, dual_socp.ub, dual_socp.lb,
                                 dual_socp.qmat, [], dual_socp.obj)
                self.dual = formula
                return formula

            if len(primal.qmat) == 0:
                pxmat = primal.xmat
            else:
                num_exp = len(primal.xmat)
                i_idx = np.array([range(num_exp)] * 3).T.flatten()
                j_idx = flat(primal.xmat)
                sp_xmat = sp.csr_matrix(([1, 1, 1] * num_exp,
                                         (i_idx, j_idx)),
                                        (num_exp, primal.linear.shape[1]))
                socp_idx = flat(primal.qmat)
                keep_idx = [i for i in range(primal.linear.shape[1])
                            if i not in socp_idx]
                sp_xmat = sp_xmat[:, keep_idx]
                pxmat = [list(sp_xmat[i].indices) for i in range(num_exp)]

            # eye_indices = [item for inner in pxmat for item in inner]
            # eye_block = dual_socp.linear[eye_indices, :]
            # if len(eye_block.data) + 1 == len(eye_block.indptr):

            linear = dual_socp.linear
            const = dual_socp.const
            sense = dual_socp.sense
            obj = dual_socp.obj
            vtype = dual_socp.vtype
            ub = dual_socp.ub
            lb = dual_socp.lb
            qmat = dual_socp.qmat

            num_xc = len(pxmat)
            count_col = np.arange(0, 3*num_xc, 3).reshape((num_xc, 1))
            xmat = count_col + np.array([[0, 1, 2]]*num_xc)
            xmat = list((linear.shape[1] + xmat))
            data = [-1, 1, -1, -1] * num_xc
            i_idx = np.array([ex + ex[-1:] for ex in pxmat]).flatten()
            j_idx = (count_col + np.array([2, 1, 0, 2])).flatten()
            extra_block = sp.csr_matrix((data, (i_idx, j_idx)),
                                        (linear.shape[0], 3*num_xc))
            linear = sp.hstack((linear, extra_block))
            obj = np.hstack((obj, np.zeros(3*num_xc)))
            ub = np.hstack((ub, np.inf*np.ones(3*num_xc)))
            lb = np.hstack((lb, -np.inf*np.ones(3*num_xc)))
            vtype = np.hstack((vtype, np.array(['C']*3*num_xc)))
            formula = GCProg(sp.csr_matrix(linear), const, sense,
                             vtype, ub, lb, qmat, xmat, obj)

            self.dual = formula
            self.dupdate = False
            return formula


class GCProg(SOCProg):

    def __init__(self, linear, const, sense, vtype, ub, lb, qmat, xmat, obj=None):

        super().__init__(linear, const, sense, vtype, ub, lb, qmat, obj)
        self.xmat = xmat

    def __repr__(self):

        xmat = self.xmat
        string = 'Conic program object:\n'
        socp_str = super().__repr__()
        string += socp_str[socp_str.find('\n')+1:]
        string += '---------------------------------------------\n'
        string += 'Number of ExpCone constraints: {0}\n'.format(len(xmat))

        return string

    def showec(self):

        n = len(self.xmat)

        if n == 0:
            return None

        indices = np.concatenate([item for item in self.xmat])
        values = [1, 2, 3] * n
        indptr = range(0, 3*n+1, 3)

        var_names = ['x{0}'.format(i)
                     for i in range(1, self.linear.shape[1] + 1)]
        constr_names = ['EC{0}'.format(j)
                        for j in range(1, n + 1)]
        table = pd.DataFrame(sp.csr_matrix((values, indices, indptr),
                             (n, self.linear.shape[1])).todense(),
                             index=constr_names, columns=var_names)
        table['sense'] = ['-'] * n
        table['constant'] = ['-'] * n

        return table

    def show(self):
        """
        Returns a pandas.DataFrame that summarizes the information on the
        optimization problem.
        """

        table = self.showlc()
        obj_row = pd.DataFrame(self.obj.reshape((1, self.obj.size)),
                               columns=table.columns[:-2], index=['Obj'])
        table = pd.concat([obj_row, table], axis=0)

        table_qc = self.showqc()
        if table_qc is not None:
            table = pd.concat([table, table_qc], axis=0)
        table_ec = self.showec()
        if table_ec is not None:
            table = pd.concat([table, table_ec], axis=0)

        ub = pd.DataFrame(self.ub.reshape((1, self.ub.size)),
                          columns=table.columns[:-2], index=['UB'])
        lb = pd.DataFrame(self.lb.reshape((1, self.lb.size)),
                          columns=table.columns[:-2], index=['LB'])
        vtype = pd.DataFrame(self.vtype.reshape((1, self.vtype.size)),
                             columns=table.columns[:-2], index=['Type'])
        table = pd.concat([table, ub, lb, vtype], axis=0)

        return table.fillna('-')

from .lp import Affine, CvxConstr, PCvxConstr, ExpConstr, KLConstr, LMIConstr, IPCone
from .lp import def_sol, Solution
from .lp import rstack
from .socp import Model as SOCModel
from .socp import SOCProg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections.abc import Iterable
from .subroutines import flat, rso_broadcast, vert_comb


class Model(SOCModel):
    """
    The Model class creates an SOCP model object
    """

    def __init__(self, nobj=False, mtype='R', name=None, top=None):

        super().__init__(nobj, mtype, name, top)
        self.psd_vars = []
        self.exp_constr = []
        self.other_constr = []
        self.det_constr = []

    def reset(self):

        self.lin_constr = []
        self.pws_constr = []
        self.cone_constr = []
        self.exp_constr = []
        self.other_constr = []
        self.bounds = []
        self.aux_constr = []
        self.aux_bounds = []
        self.aux_ipc = []
        self.cvx_constr = []

    def st(self, constr):

        if isinstance(constr, Iterable):
            for item in constr:
                self.st(item)
        elif isinstance(constr, ExpConstr):
            self.exp_constr.append(constr)
        elif isinstance(constr, CvxConstr):
            if constr.xtype in 'XLPFN':
                self.other_constr.append(constr)
            elif constr.xtype in 'OD':
                self.det_constr.append(constr)
            else:
                super().st(constr)
        elif isinstance(constr, KLConstr):
            self.other_constr.append(constr)
        elif isinstance(constr, LMIConstr):
            self.other_constr.append(constr)
        else:
            super().st(constr)

        self.pupdate = True
        self.dupdate = True

        return constr

    def do_math(self, primal=True, refresh=True, obj=True):
        """
        Return the linear or conic programming problem as the
        standard formula of the model.

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
            A conic programming problem.
        """

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal

            if refresh:
                self.auxs = []
                self.aux_constr = []
                self.aux_bounds = []
                self.aux_ipc = []
                self.last = self.vars[-1].first + self.vars[-1].size

            more_exp = []
            more_others = []
            more_det = []
            if self.obj is not None:
                obj_constr = (self.vars[0] - self.sign * self.obj >= 0)
                if isinstance(obj_constr, CvxConstr):
                    constr = obj_constr
                    if constr.xtype in 'XLPF':
                        more_others.append(constr)
                    elif constr.xtype in 'OD':
                        more_det.append(constr)

            for constr in self.det_constr + more_det:
                if constr.xtype == 'O':
                    affine_in = constr.affine_in
                    affine_out = constr.affine_out * (1/constr.multiplier)
                    dim = affine_in.shape[0]

                    Zmat = self.dvar((dim, dim), aux=True).to_affine().tril()
                    vec = self.dvar(dim, aux=True)

                    self.aux_constr.append(vec.sum() - affine_out >= 0)
                    more_others.append(vec <= Zmat.diag().log())
                    more_others.append(rstack([affine_in, Zmat],
                                              [Zmat.T, Zmat.diag(fill=True)]) >> 0)
                    more_others.append(affine_in >> 0)
                elif constr.xtype == 'D':
                    affine_in = constr.affine_in
                    affine_out = constr.affine_out * (1/constr.multiplier)
                    dim = affine_in.shape[0]

                    Zmat = self.dvar((dim, dim), aux=True).to_affine().tril()
                    val = self.dvar(1, aux=True)

                    self.aux_constr.append(val - affine_out >= 0)
                    self.aux_ipc.append(IPCone(val, Zmat.diag(), [1]*dim))
                    more_others.append(rstack([affine_in, Zmat],
                                              [Zmat.T, Zmat.diag(fill=True)]) >> 0)
                    more_others.append(affine_in >> 0)

            lmi = []
            for constr in self.other_constr + more_others:
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
                                                        exprs[0],
                                                        -exprs[2], exprs[1])
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
                    elif constr.xtype == 'F':
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        exprs_list = rso_broadcast(constr.affine_in, affine_out)
                        ns = len(exprs_list)
                        aux_var = self.dvar((ns, 2))
                        self.aux_constr.append(aux_var.sum(axis=1) <= 1)
                        for s, exprs in enumerate(exprs_list):
                            exp_cone_constr = ExpConstr(constr.model,
                                                        exprs[0] + exprs[1],
                                                        aux_var[s, 0], 1)
                            self.exp_constr.append(exp_cone_constr)
                            exp_cone_constr = ExpConstr(constr.model,
                                                        exprs[1],
                                                        aux_var[s, 1], 1)
                            self.exp_constr.append(exp_cone_constr)
                    elif constr.xtype == 'N':
                        affine_in = constr.affine_in
                        affine_out = constr.affine_out * (1/constr.multiplier)
                        order = constr.params
                        dim_in = affine_in.size
                        aux_xvar = self.dvar(dim_in).to_affine()
                        aux_zvar = self.dvar(dim_in).to_affine()
                        aux_rvar = self.dvar(dim_in).to_affine()
                        aux_yvar = self.dvar().to_affine()
                        self.aux_constr.append(affine_in <= aux_xvar)
                        self.aux_constr.append(-affine_in <= aux_xvar)
                        self.aux_constr.append(aux_zvar.sum() <= aux_yvar)
                        self.aux_constr.append(aux_yvar + affine_out <= 0)
                        for s in range(dim_in):
                            exp_cone_constr = ExpConstr(constr.model,
                                                        -aux_rvar[s] * (1/(order - 1)),
                                                        aux_yvar,
                                                        aux_xvar[s])
                            self.exp_constr.append(exp_cone_constr)
                            exp_cone_constr = ExpConstr(constr.model,
                                                        aux_rvar[s],
                                                        aux_zvar[s],
                                                        aux_xvar[s])
                            self.exp_constr.append(exp_cone_constr)
                elif isinstance(constr, LMIConstr):
                    lmi.append({'linear': constr.linear,
                                'const': constr.const,
                                'dim': constr.dim})
                    expr = Affine(constr.model, constr.linear, constr.const)
                    dim = expr.shape[0]
                    tridx = np.triu_indices(dim, 1)
                    self.aux_constr.append(expr[tridx] == expr.T[tridx])

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
                             formula.qmat, xmat, lmi, formula.obj)
            self.primal = formula
            self.pupdate = False

            return formula

        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

            primal = self.do_math(obj=obj)
            pvar_num = primal.linear.shape[1]

            dual_socp = super().do_math(primal=False, refresh=False, obj=obj)

            if len(primal.xmat) == 0 and len(primal.lmi) == 0:
                formula = GCProg(dual_socp.linear, dual_socp.const, dual_socp.sense,
                                 dual_socp.vtype, dual_socp.ub, dual_socp.lb,
                                 dual_socp.qmat, [], [], dual_socp.obj)
                self.dual = formula
                return formula

            if len(primal.qmat) == 0:
                pxmat = primal.xmat
                plmi = primal.lmi
            else:
                num_exp = len(primal.xmat)
                i_idx = np.array([range(num_exp)] * 3).T.flatten()
                j_idx = flat(primal.xmat)
                sp_xmat = sp.csr_matrix(([1, 1, 1] * num_exp,
                                         (i_idx, j_idx)),
                                        (num_exp, pvar_num))
                socp_idx = flat(primal.qmat)
                keep_idx = [i for i in range(pvar_num)
                            if i not in socp_idx]
                sp_xmat = sp_xmat[:, keep_idx]
                pxmat = [list(sp_xmat[i].indices) for i in range(num_exp)]

                for each in primal.lmi:
                    if each['linear'].shape[1] < pvar_num:
                        each['linear'].resize((each['linear'].shape[0], pvar_num))
                    each['linear'] = each['linear'][:, keep_idx]
                plmi = primal.lmi

            linear = dual_socp.linear
            const = dual_socp.const
            sense = dual_socp.sense
            obj = dual_socp.obj
            vtype = dual_socp.vtype
            ub = dual_socp.ub
            lb = dual_socp.lb
            qmat = dual_socp.qmat

            num_xc = len(pxmat)
            if num_xc > 0:
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
            else:
                xmat = []

            num_lmi = len(plmi)
            if num_lmi > 0:
                linear_list = []
                const_list = [obj]
                dims = []
                lmi = []
                here = 0
                total_col = linear.shape[1] + sum([each['dim']**2 for each in plmi])
                for each in plmi:
                    each_linear = each['linear']
                    if each_linear.shape[1] != linear.shape[0]:
                        each_linear.resize([each_linear.shape[0], linear.shape[0]])
                    linear_list.append(each_linear)

                    each_const = -each['const']
                    const_list.append(each_const.flatten())

                    each_dim = each['dim']
                    dims.append(each['dim'])

                    each_row = each_dim ** 2
                    each_idx = np.arange(each_row)
                    temp_idx = linear.shape[1] + here + each_idx
                    lmi_linear = sp.csr_matrix((np.ones(each_row),
                                                (each_idx, temp_idx)),
                                               (each_row, total_col))
                    here += each_row
                    lmi.append({'linear': lmi_linear,
                                'const': np.zeros((each_dim, each_dim)),
                                'dim': each_dim})

                extra_block = sp.csr_matrix(sp.vstack(linear_list).T)

                linear = sp.hstack((linear, extra_block))
                obj = np.hstack((const_list))
                extra_dim = np.square(dims).sum()
                ub = np.hstack((ub, np.inf*np.ones(extra_dim)))
                lb = np.hstack((lb, -np.inf*np.ones(extra_dim)))
                vtype = np.hstack((vtype, np.array(['C']*extra_dim)))
            else:
                lmi = []

            formula = GCProg(sp.csr_matrix(linear), const, sense,
                             vtype, ub, lb, qmat, xmat, lmi, obj)

            self.dual = formula
            self.dupdate = False
            return formula

    def soc_solve(self, solver=None, degree=4, cuts=(-30, 60),
                  display=True, params={}):
        """
        Solve the approximated SOC model with the selected solver interface.

        Parameters
        ----------
            solver : {None, lpg_solver, clp_solver, ort_solver, eco_solver
                      cpx_solver, grb_solver, msk_solver}
                Solver interface used for model solution. Use default solver
                if solver=None.
            degree : int
                The L-degree value for approximating exponential cone
                constraints.
            cuts : tuple of two integers
                The lower and upper cut-off values for the SOC approximation
                of exponential constraints.
            display : bool
                Display option of the solver interface.
            params : dict
                A dictionary that specifies parameters of the selected solver.
                So far the argument only applies to Gurobi, CPLEX, and Mosek.
        """

        formula = self.do_math(obj=True).to_socp(degree, cuts)
        if solver is None:
            solution = def_sol(formula, display, params)
        else:
            solution = solver.solve(formula, display, params)

        if isinstance(solution, Solution):
            self.solution = solution
        else:
            self.solution = None


class GCProg(SOCProg):
    """
    The GCProg class creates an exponential/semidefinite cone program
    """

    def __init__(self, linear, const, sense, vtype, ub, lb, qmat, xmat, lmi, obj=None):

        super().__init__(linear, const, sense, vtype, ub, lb, qmat, obj)
        self.xmat = xmat
        self.lmi = lmi

    def __repr__(self):

        xmat = self.xmat
        lmi = self.lmi
        string = 'Conic program object:\n'
        socp_str = super().__repr__()
        string += socp_str[socp_str.find('\n')+1:]
        string += '---------------------------------------------\n'
        string += 'Number of ExpCone constraints: {0}\n'.format(len(xmat))
        string += '---------------------------------------------\n'
        string += 'Number of PSCone constraints:  {0}\n'.format(len(lmi))

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

    def showlmi(self):

        n = len(self.lmi)
        if n == 0:
            return None

        linear_list = []
        num_col = self.linear.shape[1]
        for c in self.lmi:
            each_linear = c['linear']
            if each_linear.shape[1] < num_col:
                each_linear.resize((each_linear.shape[0], num_col))
            linear_list.append(each_linear)

        left = sp.vstack(linear_list).toarray()

        right = np.concatenate([c['const'].flatten() for c in self.lmi])

        var_names = ['x{0}'.format(i)
                     for i in range(1, self.linear.shape[1] + 1)]

        constr_names = np.concatenate([[f'PSDC{i+1}'] * (c['dim'] ** 2)
                                       for i, c in enumerate(self.lmi)])

        table = pd.DataFrame(left, columns=var_names, index=constr_names)

        table['sense'] = '>>'
        table['constant'] = right

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
        table_lmi = self.showlmi()
        if table_lmi is not None:
            table = pd.concat([table, table_lmi], axis=0)

        ub = pd.DataFrame(self.ub.reshape((1, self.ub.size)),
                          columns=table.columns[:-2], index=['UB'])
        lb = pd.DataFrame(self.lb.reshape((1, self.lb.size)),
                          columns=table.columns[:-2], index=['LB'])
        vtype = pd.DataFrame(self.vtype.reshape((1, self.vtype.size)),
                             columns=table.columns[:-2], index=['Type'])
        table = pd.concat([table, ub, lb, vtype], axis=0)

        return table.fillna('-')

    def to_socp(self, degree=4, cuts=(-30, 60)):

        num_vars = 1 + 4 + degree + 3
        num_cols = num_vars + (3 + degree)*3

        t_idx = 0
        x_idx = [1, 2]
        alpha_idx = [3, 4]
        fgh_idx = [5, 6, 7]
        v_idx = list(range(8, 8 + degree))

        data = [1]
        row_idx = [0]
        col_idx = [0]
        row_count = 1

        data += [1]*4
        row_idx += [1, 1, 2, 2]
        col_idx += x_idx + alpha_idx
        row_count += 2

        data += [20/2**degree/24, 23/24, 0.25, 1/24, -1]
        row_idx += [row_count]*5
        col_idx += [x_idx[1], alpha_idx[1], fgh_idx[0], fgh_idx[2], v_idx[0]]
        row_count += 1

        cut_lower, cut_upper = cuts
        data += [1, -cut_lower, 1, -cut_upper, -1, cut_lower]
        row_idx += list(np.array([0, 0, 1, 1, 2, 2]) + row_count)
        col_idx += [x_idx[0], alpha_idx[0], x_idx[1],
                    alpha_idx[1], x_idx[1], alpha_idx[1]]
        row_count += 3

        here = num_vars
        ones = [-1, -1, 1]
        data += [0.5, -0.5, 1/2**degree, -0.5, -0.5] + ones
        row_idx += list(np.array([0, 0, 1, 2, 2] + list(range(3))) + row_count)
        col_idx += [alpha_idx[1], fgh_idx[0], x_idx[1],
                    alpha_idx[1], fgh_idx[0]] + list(range(here, here+3))
        here += 3
        row_count += 3

        data += [0.5, -0.5, 1/2**degree, 1, -0.5, -0.5] + ones
        row_idx += list(np.array([0, 0, 1, 1, 2, 2] + list(range(3))) + row_count)
        col_idx += [alpha_idx[1], fgh_idx[1], x_idx[1], alpha_idx[1],
                    alpha_idx[1], fgh_idx[1]] + list(range(here, here+3))
        here += 3
        row_count += 3

        data += [0.5, -0.5, 1, -0.5, -0.5] + ones
        row_idx += list(np.array([0, 0, 1, 2, 2] + list(range(3))) + row_count)
        col_idx += [alpha_idx[1], fgh_idx[2], fgh_idx[1],
                    alpha_idx[1], fgh_idx[2]] + list(range(here, here+3))
        here += 3
        row_count += 3

        for d in range(degree - 1):
            data += [0.5, -0.5, 1, -0.5, -0.5] + ones
            row_idx += list(np.array([0, 0, 1, 2, 2] + list(range(3))) + row_count)
            col_idx += [alpha_idx[1], v_idx[d+1], v_idx[d],
                        alpha_idx[1], v_idx[d+1]] + list(range(here, here+3))
            here += 3
            row_count += 3

        data += [0.5, -0.5, 1, -0.5, -0.5] + ones
        row_idx += list(np.array([0, 0, 1, 2, 2] + list(range(3))) + row_count)
        col_idx += [alpha_idx[1], t_idx, v_idx[-1],
                    alpha_idx[1], t_idx] + list(range(here, here+3))
        here += 3
        row_count += 3

        more_linear = sp.csr_matrix((data, (row_idx, col_idx)),
                                    shape=(row_count, num_cols))
        more_sense = [0] + [1]*2 + [0]*4 + [1, 1, 0]*(3 + degree)
        more_const = np.zeros(row_count)
        more_lb = - np.ones(num_cols) * np.inf
        more_lb[alpha_idx + fgh_idx + v_idx] = 0
        for d in range(3 + degree):
            more_lb[num_vars + d*3 + 2] = 0

        left_width = self.linear.shape[1]
        qmat = self.qmat
        lmi = self.lmi

        linear = self.linear
        const = self.const
        sense = self.sense
        ub = self.ub
        lb = self.lb
        obj = self.obj
        vtype = self.vtype
        right_width = more_linear.shape[1]
        for xm in self.xmat:
            left_width = linear.shape[1]
            # print(left_width)
            left_linear = sp.csr_matrix((-np.ones(3),
                                         (range(3), [xm[1], xm[0], xm[2]])),
                                        shape=(row_count, left_width))
            linear = vert_comb(linear, sp.hstack((left_linear, more_linear)))
            # print(linear.shape)
            const = np.concatenate((const, more_const))
            sense = np.concatenate((sense, more_sense))
            ub = np.concatenate((ub, np.ones(right_width) * np.inf))
            lb = np.concatenate((lb, more_lb))
            obj = np.concatenate((obj, np.zeros(right_width)))
            vtype = np.concatenate((vtype, np.array(['C']*right_width)))

            qmat += [list(left_width + num_vars + np.array([2, 1, 0]) + q*3)
                     for q in range(3+degree)]

        # return SOCProg(linear, const, sense, vtype, ub, lb, qmat, obj)
        return GCProg(linear, const, sense, vtype, ub, lb, qmat, [], lmi, obj)

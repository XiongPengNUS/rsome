from .lp import Model as LPModel
from .lp import LinConstr, Bounds, CvxConstr, ConeConstr
from .lp import LinProg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections.abc import Iterable


class Model(LPModel):
    """
    The Model class creates an SOCP model object
    """

    def __init__(self, nobj=False, mtype='R', name=None, top=None):

        super().__init__(nobj, mtype, name, top)
        self.cvx_constr = []
        self.cone_constr = []
        # self.avar_indices = []

    def reset(self):

        self.lin_constr = []
        self.pws_constr = []
        self.cone_constr = []
        self.bounds = []
        self.aux_constr = []
        self.aux_bounds = []
        self.cvx_constr = []

    def st(self, constr):
        """
        Define constraints that an optimization model subject to.

        Parameters
        ----------
        constr
            Constraints or collections of constraints that the model
            subject to.
        """

        if isinstance(constr, Iterable):
            for item in constr:
                self.st(item)
        elif isinstance(constr, (LinConstr, Bounds)):
            super().st(constr)
        elif isinstance(constr, CvxConstr):
            if constr.model is not self:
                raise ValueError('Constraints are not defined for this model.')
            if constr.xtype in 'AMI':
                super().st(constr)
            elif constr.xtype in 'ESQ':
                self.cvx_constr.append(constr)
            else:
                raise ValueError('Unsupported convex constraints.')
        elif isinstance(constr, ConeConstr):
            if constr.model is not self:
                raise ValueError('Constraints are not defined for this model.')
            self.cone_constr.append(constr)
        else:
            raise TypeError('Unknown constraint type.')

        self.pupdate = True
        self.dupdate = True

    def do_math(self, primal=True, refresh=True, obj=True):
        """
        Return the linear or second-order cone programming problem
        as the standard formula of the model.

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
        prog : SOCProg
            A second-order cone programming problem.
        """

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal

            if refresh:
                self.auxs = []
                self.aux_constr = []
                self.aux_bounds = []
                self.last = self.vars[-1].first + self.vars[-1].size

            more_cvx = []
            if self.obj is not None:
                obj_constr = (self.vars[0] - self.sign * self.obj >= 0)
                if isinstance(obj_constr, CvxConstr):
                    more_cvx.append(obj_constr)

            qmat = []
            for constr in self.cvx_constr + more_cvx:
                if constr.xtype == 'E':
                    aux_left = self.dvar(constr.affine_in.shape, aux=True)
                    aux_right = self.dvar(1, aux=True)
                    affine_in = constr.affine_in * constr.multiplier
                    self.aux_constr.append(affine_in - aux_left == 0)
                    self.aux_constr.append(constr.affine_out + aux_right <= 0)
                    bounds = (aux_right >= 0)
                    if isinstance(bounds, Bounds):
                        self.aux_bounds.append(bounds)
                    else:
                        self.aux_constr.append(bounds)
                    qmat.append([aux_right.first] +
                                list([aux_left.first + index
                                      for index in range(aux_left.size)]))
                elif constr.xtype == 'S':
                    aux1 = self.dvar(constr.affine_out.shape, aux=True)
                    aux2 = self.dvar(constr.affine_in.shape, aux=True)
                    aux3 = self.dvar(constr.affine_out.shape, aux=True)
                    affine_in = constr.affine_in * constr.multiplier
                    self.aux_constr.append(aux1 - 0.5*(1+constr.affine_out)
                                           == 0)
                    self.aux_constr.append(aux2 - affine_in == 0)
                    self.aux_constr.append(aux3 - 0.5 * (1-constr.affine_out)
                                           == 0)
                    bounds = (aux3 >= 0)
                    if isinstance(bounds, Bounds):
                        self.aux_bounds.append(bounds)
                    else:
                        self.aux_constr.append(bounds)
                    for i in range(constr.affine_in.size):
                        qmat.append([aux3.first + i] +
                                    [aux1.first + i, aux2.first + i])
                elif constr.xtype == 'Q':
                    # aux1 = self.dvar(constr.affine_out.shape, aux=True)
                    aux1 = self.dvar(1, aux=True)
                    aux2 = self.dvar(constr.affine_in.shape, aux=True)
                    # aux3 = self.dvar(constr.affine_out.shape, aux=True)
                    aux3 = self.dvar(1, aux=True)
                    aux4 = self.dvar(1, aux=True)
                    affine_in = constr.affine_in * constr.multiplier
                    self.aux_constr.append(aux1 - 0.5 * (1-aux4)
                                           == 0)
                    self.aux_constr.append(aux2 - affine_in == 0)
                    self.aux_constr.append(aux3 - 0.5 * (1+aux4)
                                           == 0)
                    self.aux_constr.append(aux4 + constr.affine_out <= 0)
                    bounds = (aux3 >= 0)
                    if isinstance(bounds, Bounds):
                        self.aux_bounds.append(bounds)
                    else:
                        self.aux_constr.append(bounds)
                    qmat.append([aux3.first] + [aux1.first] +
                                list(aux2.first + np.arange(aux2.size)))

            for constr in self.cone_constr:
                qmat.append([constr.right_var.first + constr.right_index] +
                            [constr.left_var.first + index
                             for index in constr.left_index])

            formula = super().do_math(primal=True, refresh=False, obj=obj)
            formula = SOCProg(formula.linear, formula.const, formula.sense,
                              formula.vtype, formula.ub, formula.lb,
                              qmat, formula.obj)
            self.primal = formula
            self.pupdate = False

            return formula
        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

            primal = self.do_math(obj=obj)

            dual_lp = super().do_math(primal=False, refresh=False, obj=obj)
            if len(primal.qmat) == 0:
                formula = SOCProg(dual_lp.linear, dual_lp.const, dual_lp.sense,
                                  dual_lp.vtype, dual_lp.ub, dual_lp.lb,
                                  [], dual_lp.obj)
                self.dual = formula
                return formula

            eye_indices = [item for inner in primal.qmat for item in inner]
            eye_block = dual_lp.linear[eye_indices, :]
            if len(eye_block.data) + 1 == len(eye_block.indptr):
                lin_indices = [ind for ind in range(primal.linear.shape[1])
                               if ind not in eye_indices]
                linear = dual_lp.linear[lin_indices, :]
                const = dual_lp.const[lin_indices]
                sense = dual_lp.sense[lin_indices]
                obj = dual_lp.obj
                vtype = dual_lp.vtype
                ub = dual_lp.ub
                lb = dual_lp.lb
                qmat = []
                for qc in primal.qmat:
                    lbz_index = dual_lp.linear[qc[0], :].indices
                    ub[lbz_index] = - lb[lbz_index]
                    lb[lbz_index] = 0
                    linear[:, lbz_index] = - linear[:, lbz_index]
                    obj[lbz_index] = - obj[lbz_index]
                    qmat.append(list(dual_lp.linear[qc, :].indices))

                formula = SOCProg(linear, const, sense,
                                  vtype, ub, lb, qmat, obj)

            else:
                linear = dual_lp.linear
                num_constr = dual_lp.linear.shape[1]
                extra_nvar = len(eye_indices)
                extra_block = sp.csr_matrix((np.ones(extra_nvar),
                                             (eye_indices, np.arange(extra_nvar))),
                                            shape=(linear.shape[0], extra_nvar))
                linear = sp.csr_matrix(sp.hstack((linear, extra_block)))
                const = dual_lp.const
                sense = dual_lp.sense

                obj = np.concatenate((dual_lp.obj, np.zeros(extra_nvar)))
                vtype = np.concatenate((dual_lp.vtype, np.array(['C']*extra_nvar)))

                ub = np.concatenate((dual_lp.ub, np.ones(extra_nvar)*np.infty))
                extra_lb = - np.ones(extra_nvar)*np.infty
                ind_pos = 0
                for qc in primal.qmat:
                    extra_lb[ind_pos] = 0
                    ind_pos += len(qc)
                lb = np.concatenate((dual_lp.lb, extra_lb))

                qmat = []
                ind_pos = 0
                for qc in primal.qmat:
                    qmat.append([ind_pos+num_constr] +
                                [k+ind_pos+num_constr for k in range(1, len(qc))])
                    ind_pos += len(qc)

                formula = SOCProg(linear, const, sense,
                                  vtype, ub, lb, qmat, obj)

            self.dual = formula
            self.dupdate = False

            return formula


class SOCProg(LinProg):
    """
    The SOCProg class creates an second-order cone program
    """

    def __init__(self, linear, const, sense, vtype, ub, lb, qmat, obj=None):

        super().__init__(linear, const, sense, vtype, ub, lb, obj)
        self.qmat = qmat

    def __repr__(self):

        qmat = self.qmat
        string = 'Second order cone program object:\n'
        string += super().__repr__()
        string += '---------------------------------------------\n'
        string += 'Number of SOC constraints:     {0}\n'.format(len(qmat))

        return string

    def showqc(self):

        n = len(self.qmat)

        if n == 0:
            return None

        indices = np.concatenate([item for item in self.qmat])
        values = np.concatenate([[-1.0] + [1.0]*(len(item)-1)
                                 for item in self.qmat])
        indptr = [0] * (n + 1)
        for i in range(n):
            indptr[i+1] = indptr[i] + len(self.qmat[i])

        var_names = ['x{0}'.format(i)
                     for i in range(1, self.linear.shape[1] + 1)]
        constr_names = ['QC{0}'.format(j)
                        for j in range(1, n + 1)]
        table = pd.DataFrame(sp.csr_matrix((values, indices, indptr),
                             (n, self.linear.shape[1])).todense(),
                             index=constr_names, columns=var_names)
        table['sense'] = ['<='] * n
        table['constant'] = [0.0] * n

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

        ub = pd.DataFrame(self.ub.reshape((1, self.ub.size)),
                          columns=table.columns[:-2], index=['UB'])
        lb = pd.DataFrame(self.lb.reshape((1, self.lb.size)),
                          columns=table.columns[:-2], index=['LB'])
        vtype = pd.DataFrame(self.vtype.reshape((1, self.vtype.size)),
                             columns=table.columns[:-2], index=['Type'])
        table = pd.concat([table, ub, lb, vtype], axis=0)

        return table.fillna('-')

    def lp_export(self):

        string = super().lp_export()
        index_st = string.find('Subject To')
        s1 = string[:index_st+11]
        s2 = string[index_st+11:]
        sq = ''
        for i, qc in enumerate(self.qmat):
            sq += ' q{}: [ '.format(i+1)
            sq += ' + '.join(['x{} ^2'.format(j+1) for j in qc[1:]])
            sq += ' - x{} ^2 ] <= 0\n'.format(qc[0]+1)

        return s1 + sq + s2

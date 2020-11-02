from .socp import Model as SOCModel
from .ro import Model as ROModel
from .lp import LinConstr, ConeConstr, CvxConstr
from .lp import Vars, Affine
from .lp import RoAffine, RoConstr
from .lp import DecVar, RandVar, DecLinConstr, DecCvxConstr
from .lp import DecRoConstr
from .lp import Scen
from .subroutines import *
import numpy as np
import pandas as pd
from collections import Sized, Iterable


class Model:
    """
    Returns a model object with the given number of scenarios.

    Parameters
    ----------
    scens : int or array-like objects
        The number of scenarios, if it is an integer. It could also be
        an array of scenario indices.
    name : str
        Name of the model

    Returns
    -------
    model : rsome.dro.Model
        A model object
    """

    def __init__(self, scens=1, name=None):

        self.ro_model = ROModel()
        self.vt_model = SOCModel(mtype='V')
        self.sup_model = self.ro_model.sup_model
        self.exp_model = SOCModel(nobj=True, mtype='E')
        self.pro_model = SOCModel(nobj=True, mtype='P')

        self.obj_ambiguity = None

        if isinstance(scens, int):
            num_scen = scens
            series = pd.Series(np.arange(num_scen).astype(int))
        elif isinstance(scens, Sized):
            num_scen = len(scens)
            series = pd.Series(np.arange(num_scen).astype(int), index=scens)
        else:
            raise ValueError('Incorrect scenarios.')
        self.num_scen = num_scen
        self.series_scen = series

        self.dec_vars = [DecVar(self, self.vt_model.vars[0])]
        self.rand_vars = []
        self.all_constr = []
        self.var_ev_list = None
        # self.affadapt_mat = None

        pr = self.pro_model.dvar(num_scen, name='probabilities')
        self.p = pr

        self.obj = None
        self.sign = 1

        self.primal = None
        self.dual = None
        self.solution = None
        self.pupdate = True
        self.dupdate = True

        self.solution = None

        self.name = name

    def rvar(self, shape=(1,), name=None):

        sup_var = self.sup_model.dvar(shape, 'C', name)
        exp_var = self.exp_model.dvar(shape, 'C', name)
        rand_var = RandVar(sup_var, exp_var)
        self.rand_vars.append(rand_var)

        return rand_var

    def dvar(self, shape=(1,), vtype='C', name=None):
        """
        Returns an array of decision variables with the given shape
        and variable type.

        Parameters
        ----------
        shape : int or tuple
            Shape of the variable array.
        vtype : {'C', 'B', 'I'}
            Type of the decision variables. 'C' means continuous; 'B'
            means binary, and 'I" means integer.
        name : str
            Name of the variable array

        Returns
        -------
        new_var : rsome.lp.DecVar
            An array of new decision variables
        """

        dec_var = self.vt_model.dvar(shape, vtype, name)
        dec_var = DecVar(self, dec_var, name=name)
        self.dec_vars.append(dec_var)

        return dec_var

    def ambiguity(self):

        if self.all_constr:
            raise SyntaxError('Ambiguity set must be specified ' +
                              'before defining constraints.')

        return Ambiguity(self)

    def bou(self, *args):

        if self.num_scen != 1:
            raise ValueError('The uncertainty set can only be applied '
                             'to a one-scenario model')

        bou_set = self.ambiguity()
        for arg in args:
            if arg.model is not self.sup_model:
                raise ValueError('Constraints are not for this support.')

        bou_set.sup_constr = [tuple(args)]

        return bou_set

    def wks(self, *args):

        if self.num_scen != 1:
            raise ValueError('The WKS ambiguity set can only be applied '
                             'to a one-scenario model')

        wks_set = self.ambiguity()
        sup_constr = []
        exp_constr = []
        for arg in args:
            if arg.model is self.sup_model:
                sup_constr.append(arg)
            elif arg.model is self.exp_model:
                exp_constr.append(arg)
            else:
                raise ValueError('Constraints are not defined for the '
                                 'ambiguity support.')

        wks_set.sup_constr = [tuple(sup_constr)]
        wks_set.exp_constr = [tuple(exp_constr)]
        wks_set.exp_constr_indices = [np.array([0], dtype=np.int32)]

        return wks_set

    def rule_var(self):

        if self.var_ev_list is not None:
            return self.var_ev_list

        total = sum(dvar.size*len(dvar.event_adapt)
                    for dvar in self.dec_vars)
        vtype = ''.join([dvar.vtype * dvar.size * len(dvar.event_adapt)
                         if len(dvar.vtype) == 1
                         else dvar.vtype * len(dvar.event_adapt)
                         for dvar in self.dec_vars])
        var_const = self.ro_model.dvar(total, vtype=vtype)

        count = 0
        for dvar in self.dec_vars:
            dvar.ro_first = count
            count += dvar.size*len(dvar.event_adapt)

        num_scen = self.num_scen
        self.var_ev_list = []
        for s in range(num_scen):
            start = 0
            index = []
            total_size = 0
            total_col = 0
            for dvar in self.dec_vars:
                edict = event_dict(dvar.event_adapt)
                size = dvar.size
                index.extend(list(start + size * edict[s]
                                  + np.arange(size, dtype=int)))

                start += size * len(dvar.event_adapt)
                total_size += size
                total_col += size * len(dvar.event_adapt)
            """
            tr_mat = csr_matrix(([1.0] * total_size, index,
                                 range(total_size + 1)),
                                shape=(total_size, total_col))

            self.var_ev_list.append(tr_mat @ var_const)
            """

            self.var_ev_list.append(var_const[index].to_affine())

        if self.sup_model.vars:

            adapt_list = [dvar.rand_adapt if dvar.rand_adapt is not None else
                          np.zeros((dvar.size, self.sup_model.vars[-1].last))
                          for dvar in self.dec_vars]
            depend_mat = np.concatenate(adapt_list, axis=0)
            if depend_mat.sum() > 0:

                num_depend = np.array([int(dvar.rand_adapt.sum())
                                       if dvar.rand_adapt is not None else 0
                                       for dvar in self.dec_vars])
                scen_depend = np.array([len(dvar.event_adapt)
                                        for dvar in self.dec_vars])
                total_depend = num_depend @ scen_depend
                each_depend = num_depend.sum()
                var_linear = self.ro_model.dvar(total_depend)
                nz_rows = np.where(depend_mat.flatten())[0]
                for s in range(num_scen):
                    start = 0
                    index = []
                    total_var = 0
                    for dvar, num, scen in zip(self.dec_vars,
                                               num_depend, scen_depend):
                        if num == 0:
                            continue
                        edict = event_dict(dvar.event_adapt)
                        index.extend(list(start + num * edict[s]
                                     + np.arange(num, dtype=int)))
                        start += num * scen
                        total_var += num

                    depend_affine = var_linear[index].to_affine()

                    nz_cols = depend_affine.linear.indices
                    last = depend_affine.linear.shape[1]

                    ra_linear = sp.csr_matrix(([1.0] * each_depend,
                                               (nz_rows, nz_cols)),
                                              shape=(depend_mat.size, last))
                    ra_const = np.zeros(depend_mat.shape)
                    raffine = Affine(self.ro_model.rc_model,
                                     ra_linear, ra_const)
                    self.var_ev_list[s] = RoAffine(raffine,
                                                   self.var_ev_list[s],
                                                   self.ro_model.sup_model)

        return self.var_ev_list

    def min(self, obj):

        if obj.size > 1:
            raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.sign = 1
        self.pupdate = True
        self.dupdate = True

    def max(self, obj):

        if obj.size > 1:
            raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def minsup(self, obj, ambset):

        if np.prod(obj.shape) > 1:
            raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.obj_ambiguity = ambset
        self.sign = 1
        self.pupdate = True
        self.dupdate = True

    def maxinf(self, obj, ambset):

        if np.prod(obj.shape) > 1:
            raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.obj_ambiguity = ambset
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def st(self, *arg):

        for constr in arg:
            if isinstance(constr, Iterable):
                for item in constr:
                    self.st(item)
            else:
                self.all_constr.append(constr)

    def do_math(self, primal=True):

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal
        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

        self.ro_model.reset()
        self.rule_var()

        # Event-wise objective function
        self.ro_model.min(self.ro_model.rc_model.vars[1][0].to_affine())
        sign = self.sign
        constr = (self.dec_vars[0] >= self.obj * sign)
        if isinstance(constr, DecCvxConstr):
            ro_constr_list = self.ro_to_roc(constr)
        elif constr.ctype == 'R':
            ro_constr_list = self.ro_to_roc(constr)
        elif constr.ctype == 'E':
            ro_constr_list = self.dro_to_roc(constr)
        else:
            raise SyntaxError('Syntax error.')

        self.ro_model.st(ro_constr_list)

        # Event-wise Constraints
        for constr in self.all_constr:
            if isinstance(constr, DecCvxConstr):
                ro_constr_list = self.ro_to_roc(constr)
            elif constr.ctype == 'R':
                ro_constr_list = self.ro_to_roc(constr)
            elif constr.ctype == 'E':
                ro_constr_list = self.dro_to_roc(constr)
            else:
                raise SyntaxError('Syntax error')
            self.ro_model.st(ro_constr_list)

        formula = self.ro_model.do_math(primal)
        if primal:
            self.primal = formula
            self.pupdate = False
        else:
            self.dual = formula
            self.dupdate = False

        return formula

    def ro_to_roc(self, constr):

        drule_list = self.rule_var()
        num_var = self.vt_model.vars[-1].last

        ro_constr = []
        for event in constr.event_adapt:
            drule = drule_list[event[0]]
            if isinstance(constr, DecRoConstr):
                raf_linear, aff_linear = (constr.raffine.linear[:, :num_var],
                                          constr.affine.linear[:, :num_var])

                row_ind = np.unique(raf_linear.indices)
                if isinstance(drule, RoAffine):
                    if len(row_ind) > 0:
                        if (drule.raffine[row_ind].linear.nnz > 0 or
                                np.any(drule.raffine[row_ind].const)):
                            raise SyntaxError('Incorrect affine expressions.')

                    raffine = raf_linear @ drule.affine
                    raffine = (raffine.reshape(constr.raffine.shape)
                               + constr.raffine.const)
                    roaffine = RoAffine(raffine, np.zeros(aff_linear.shape[0]),
                                        self.sup_model)
                    roaffine = roaffine + aff_linear@drule
                    ew_constr = RoConstr(roaffine, constr.sense)

                elif isinstance(drule, Affine):
                    raffine = raf_linear @ drule
                    raffine = (raffine.reshape(constr.raffine.shape)
                               + constr.raffine.const)
                    roaffine = RoAffine(raffine, aff_linear@drule,
                                        self.sup_model)
                    ew_constr = RoConstr(roaffine, constr.sense)
                else:
                    raise TypeError('Unknown type.')

            elif isinstance(constr, DecLinConstr):
                linear = constr.linear
                const = constr.const
                roaffine = linear @ drule - const.reshape(const.size)
                if isinstance(roaffine, RoAffine):
                    ew_constr = RoConstr(roaffine, constr.sense)
                elif isinstance(roaffine, Affine):
                    ew_constr = LinConstr(roaffine.model, roaffine.linear,
                                          roaffine.const, constr.sense)
                else:
                    raise TypeError('Unknown type.')
            elif isinstance(constr, DecCvxConstr):
                linear_in = constr.affine_in.linear
                const_in = constr.affine_in.const
                aff_in = linear_in@drule + const_in.reshape(const_in.size)
                if isinstance(aff_in, RoAffine):
                    if (aff_in.raffine.linear.nnz > 0 or
                            np.any(aff_in.raffine.const)):
                        raise SyntaxError('Incorrect convex expressions.')
                    aff_in = aff_in.affine
                linear_out = constr.affine_out.linear
                const_out = constr.affine_out.const
                aff_out = linear_out@drule + const_out.reshape(const_out.size)
                if isinstance(aff_out, RoAffine):
                    if (aff_out.raffine.linear.nnz > 0 or
                            np.any(aff_out.raffine.const)):
                        raise SyntaxError('Incorrect convex expressions.')
                    aff_out = aff_out.affine
                ew_constr = CvxConstr(aff_in.model, aff_in, aff_out,
                                      constr.xtype)
            else:
                raise TypeError('Unknown constraint type.')

            if isinstance(ew_constr, RoConstr):
                if (ew_constr.raffine.linear.nnz > 0 or
                        np.any(ew_constr.raffine.const)):
                    if constr.ambset is None:
                        if self.obj_ambiguity is None:
                            raise SyntaxError('The Ambiguity set is '
                                              'undefined.')
                        else:
                            ambset = self.obj_ambiguity
                            support = ambset.sup_constr[event[0]]
                    else:
                        support = constr.ambset.sup_constr[event[0]]
                    ew_constr = ew_constr.forall(support)
                else:
                    ew_constr = LinConstr(ew_constr.affine.model,
                                          ew_constr.affine.linear,
                                          ew_constr.affine.const,
                                          ew_constr.sense)

            ro_constr.append(ew_constr)

        return ro_constr

    def dro_to_roc(self, constr):

        drule_list = self.rule_var()
        num_var = self.vt_model.vars[-1].last
        num_scen = self.num_scen
        num_rand = self.sup_model.vars[-1].last

        # Ambiguity set of the constr
        ambset = constr.ambset if constr.ambset else self.obj_ambiguity
        if not ambset:
            raise ValueError('The ambiguity set is undefined.')
        mixed_support = ambset.mix_support(primal=False)
        p = ambset.mix_model.vars[0][:num_scen]
        var_exp_list = ambset.mix_model.vars[1:]
        num_event = len(ambset.exp_constr)

        # Constraint components
        if isinstance(constr, DecLinConstr):
            linear = constr.linear
            const = constr.const
            raffine = None
        else:
            linear = constr.affine.linear
            const = constr.affine.const
            raffine = constr.raffine

        # Standardize constraints
        ro_constr = []
        for i in range(linear.shape[0]):
            alpha = self.ro_model.dvar(num_scen)
            if num_event:
                beta = self.ro_model.dvar((num_rand, num_event))
            else:
                beta = 0

            left = alpha @ p
            for j in range(num_event):
                left += var_exp_list[j][:num_rand] @ beta[:, j]
            ro_constr.extend((left <= 0).le_to_rc(mixed_support))

            z = Vars(self.sup_model, 0, (num_rand,), 'C', None)
            for s in range(num_scen):
                drule = drule_list[s]
                left = linear[i, :num_var] @ drule + const[i]
                if raffine:
                    if isinstance(drule, RoAffine):
                        temp = drule.affine
                        left = left.affine.reshape(left.shape)
                    elif isinstance(drule, Affine):
                        temp = drule
                    else:
                        raise TypeError('Incorrect data type.')
                    row_ind = i*num_rand + np.arange(num_rand, dtype=int)
                    new_raffine = raffine.linear[row_ind] @ temp
                    new_raffine = new_raffine.reshape((1, new_raffine.size))
                    left = RoAffine(new_raffine + raffine.const[i, :num_rand],
                                    left, constr.rand_model) ######################
                    

                event_indices = [k for k in range(num_event)
                                 if s in ambset.exp_constr_indices[k]]
                if len(event_indices) > 0:
                    right = alpha[s] + (z @ beta[:, event_indices]).sum()
                else:
                    right = alpha[s]
                inequality = (left <= right)
                if isinstance(inequality, RoConstr):
                    ro_constr.append(inequality.forall(ambset.sup_constr[s]))
                elif isinstance(inequality, LinConstr):
                    ro_constr.append(inequality)
                else:
                    raise TypeError('Incorrect data type.')

        return ro_constr

    def solve(self, solver, display=True, export=False):
        """
        Solve the model with the selected solver interface.

        Parameters
        ----------
            solver : {grb_solver, msk_solver}
                Solver interface used for model solution.
            display : bool
                Display option of the solver interface.
            export : bool
                Export option of the solver interface. A standard model file
                is generated if the option is True.
        """

        solution = solver.solve(self.do_math(), display, export)
        self.ro_model.solution = solution
        self.ro_model.rc_model.solution = solution
        self.solution = solution

    def get(self):

        if self.solution is None:
            raise SyntaxError('The model is unsolved or no feasible solution.')
        return self.sign * self.solution.objval


class Ambiguity:

    def __init__(self, model):

        self.model = model

        self.sup_constr = [None] * model.num_scen
        self.exp_constr = []
        self.exp_constr_indices = []
        self.mix_model = None

        p = self.model.p
        self.pro_constr = [p >= 0, sum(p) == 1]
        self.s = Scen(self, self.model.series_scen, self.model.p)

        self.update = True

    def __str__(self):

        return self.s.__str__()

    def __repr__(self):

        return self.s.__repr__()

    def showevents(self):

        num_scen = self.model.num_scen
        table = pd.DataFrame(['undefined']*num_scen,
                             columns=['support'],
                             index=self.s.series.index)
        sup_constr = self.sup_constr
        defined = pd.notnull(pd.Series(sup_constr,
                                       index=self.s.series.index))
        table.loc[defined, 'support'] = 'defined'

        # exp_cosntr = self.model.exp_constr
        exp_constr_indices = self.exp_constr_indices
        count = 0
        for indices in exp_constr_indices:
            column = 'expectation {0}'.format(count)
            count += 1
            table[column] = False
            table.iloc[indices, count] = True

        return table

    def __getitem__(self, indices):

        return self.s[indices]

    @property
    def loc(self):

        return self.s.loc

    @property
    def iloc(self):

        return self.s.iloc

    def suppset(self, *args):

        self.update = True
        return self.s.suppset(*args)

    def exptset(self, *args):

        self.update = True
        return self.s.exptset(*args)

    def probset(self, *args):

        self.update = True
        for arg in args:
            if arg.model is not self.model.pro_model:
                raise ValueError('Constraints are not defined for the ' +
                                 'probability set.')

        pr = self.model.p
        self.pro_constr = [pr >= 0, pr.sum() == 1] + list(args)

    def mix_support(self, primal=True):

        if not self.update and self.mix_model is not None:
            return self.mix_model.do_math(primal)

        self.model.pro_model.reset()
        self.model.pro_model.st(self.pro_constr)
        pro_support = self.model.pro_model.do_math()
        self.mix_model = SOCModel(nobj=True, mtype='M')

        # Constraints for probabilities
        p = self.mix_model.dvar(pro_support.linear.shape[1])
        constr = LinConstr(self.mix_model,
                           pro_support.linear, pro_support.const,
                           pro_support.sense)
        self.mix_model.st(constr)
        for q in pro_support.qmat:
            cconstr = ConeConstr(self.mix_model, p, q[1:], p, q[0])
            self.mix_model.st(cconstr)

        # Constraints for expectations
        for econstr, indices in zip(self.exp_constr, self.exp_constr_indices):
            self.model.exp_model.reset()
            self.model.exp_model.st(econstr)
            exp_support = self.model.exp_model.do_math()
            exp_var = self.mix_model.dvar(exp_support.linear.shape[1])
            affine = (exp_support.linear @ exp_var
                      - p[indices].sum() * exp_support.const)
            constr = LinConstr(affine.model, affine.linear, affine.const,
                               exp_support.sense)
            self.mix_model.st(constr)
            for q in exp_support.qmat:
                cconstr = ConeConstr(self.mix_model, exp_var, q[1:],
                                     exp_var, q[0])
                self.mix_model.st(cconstr)

        return self.mix_model.do_math(primal)

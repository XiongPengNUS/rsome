from .gcp import Model as GCPModel
from .ro import Model as ROModel
from .lp import DecBounds, DecExpConstr, LinConstr
from .lp import ConeConstr, PCvxConstr, CvxConstr, ExpConstr, LMIConstr
from .lp import Vars, Affine
from .lp import RoAffine, RoConstr
from .lp import DecVar, RandVar, DecLinConstr, DecCvxConstr, DecPCvxConstr
from .lp import DecRoConstr
from .lp import PiecewiseConvex, PWConstr, ExpPWConstr, DecLMIConstr
from .lp import Scen
from .lp import Solution, def_sol
from .subroutines import event_dict
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numbers import Real
from collections.abc import Sized, Iterable


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
        self.vt_model = GCPModel(mtype='V', top=self)
        self.sup_model = self.ro_model.sup_model
        self.sup_model.top = self
        self.ro_model.rc_model.top = self
        self.exp_model = GCPModel(nobj=True, mtype='E', top=self)
        self.pro_model = GCPModel(nobj=True, mtype='P', top=self)

        self.obj_ambiguity = None

        if isinstance(scens, int):
            num_scen = scens
            series = pd.Series(np.arange(num_scen).astype(int))
        elif isinstance(scens, Sized):
            num_scen = len(scens)
            series = pd.Series(np.arange(num_scen).astype(int), index=list(scens))
        else:
            raise TypeError('Incorrect scenario type.')
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

    def rvar(self, shape=(), name=None):
        """
        Returns an array of random variables with the given shape.

        Parameters
        ----------
        shape : int or tuple
            Shape of the variable array.
        name : str
            Name of the variable array

        Returns
        -------
        new_var : rsome.lp.RandVar
            An array of new random variables
        """

        sup_var = self.sup_model.dvar(shape, 'C', name)
        exp_var = self.exp_model.dvar(shape, 'C', name)
        rand_var = RandVar(sup_var, exp_var)
        self.rand_vars.append(rand_var)

        return rand_var

    def dvar(self, shape=(), vtype='C', name=None):
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
        """
        Returns an event-wise ambiguity set with the given number of scenarios
        decalred in creating the DRO model.
        """

        if self.all_constr:
            raise SyntaxError('Ambiguity set must be specified ' +
                              'before defining constraints.')

        return Ambiguity(self)

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
                index.extend(list(start + size * edict[s] +
                                  np.arange(size, dtype=int)))

                start += size * len(dvar.event_adapt)
                total_size += size
                total_col += size * len(dvar.event_adapt)

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
                        index.extend(list(start + num * edict[s] +
                                          np.arange(num, dtype=int)))
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
        """
        Minimize the given objective function.

        Parameters
        ----------
        obj
            An objective function

        Notes
        -----
        The objective function given as an array must have the size
        to be one.
        """

        if self.obj is not None:
            raise SyntaxError('Redefinition of the objective is not allowed.')

        if not isinstance(obj, (Real, PiecewiseConvex)):
            if obj.size > 1:
                raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.sign = 1
        self.pupdate = True
        self.dupdate = True

    def max(self, obj):
        """
        Maximize the given objective function.

        Parameters
        ----------
        obj
            An objective function

        Notes
        -----
        The objective function given as an array must have the size
        to be one.
        """

        if self.obj is not None:
            raise SyntaxError('Redefinition of the objective is not allowed.')

        if not isinstance(obj, (Real, PiecewiseConvex)):
            if obj.size > 1:
                raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def minsup(self, obj, ambset):
        """
        Minimize the worst-case expected objective value over the given
        ambiguity set.

        Parameters
        ----------
        obj
            Objective function involving random variables
        ambset : Ambiguity
            The ambiguity set defined for the worst-case expectation

        Notes
        -----
        The ambiguity set defined for the objective function is considered
        the default ambiguity set for the distritionally robust model.
        """

        if self.obj is not None:
            raise SyntaxError('Redefinition of the objective is not allowed.')

        if not isinstance(obj, (Real, PiecewiseConvex)):
            if obj.size > 1:
                raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.obj_ambiguity = ambset
        self.sign = 1
        self.pupdate = True
        self.dupdate = True

    def maxinf(self, obj, ambset):

        """
        Maximize the worst-case expected objective value over the given
        ambiguity set.

        Parameters
        ----------
        obj
            Objective function involving random variables
        ambset : Ambiguity
            The ambiguity set defined for the worst-case expectation

        Notes
        -----
        The ambiguity set defined for the objective function is considered
        the default ambiguity set for the distritionally robust model.
        """

        if self.obj is not None:
            raise SyntaxError('Redefinition of the objective is not allowed.')

        if not isinstance(obj, (Real, PiecewiseConvex)):
            if obj.size > 1:
                raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.obj_ambiguity = ambset
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def st(self, *arg):
        """
        Define constraints that an optimization model subject to.

        Parameters
        ----------
        *args
            Constraints or collections of constraints that the model
            subject to.

        Notes
        -----
        Multiple constraints can be defined altogether as the argument
        of the st method.
        """

        for constr in arg:
            if isinstance(constr, Iterable):
                for item in constr:
                    self.st(item)
            else:
                if isinstance(constr, (DecLinConstr, DecBounds,
                                       DecCvxConstr, DecPCvxConstr,
                                       DecExpConstr, DecLMIConstr)):
                    if constr.model is not self.vt_model:
                        raise ValueError('Models mismatch.')
                elif isinstance(constr, DecRoConstr):
                    if constr.dec_model is not self.vt_model or \
                       constr.rand_model is not self.sup_model:
                        raise ValueError('Models mismatch.')
                elif isinstance(constr, (PWConstr, ExpPWConstr)):
                    if constr.model is not self:
                        raise ValueError('Models mismatch.')
                else:
                    raise TypeError('Unsupported constraints.')

                if isinstance(constr, ExpPWConstr):
                    self.all_constr.append(constr)
                elif isinstance(constr, PWConstr):
                    self.all_constr.extend(constr.pieces)
                else:
                    self.all_constr.append(constr)

        self.pupdate = True
        self.dupdate = True

    def do_math(self, primal=True):
        """
        Return the linear, second-order cone, or exponential cone
        programming problem as the standard formula or robust
        counterpart of the model.

        Parameters
        ----------
        primal : bool, default True
            Specify whether return the primal formula of the model.
            If primal=False, the method returns the daul formula.

        Returns
        -------
        prog : GCProg
            An exponential cone programming problem.
        """

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal
        else:
            if self.dual is not None and not self.dupdate:
                return self.dual
            else:
                self.do_math(primal=True)
                return self.ro_model.do_math(False)

        self.ro_model.reset()
        self.rule_var()

        # Event-wise objective function
        self.ro_model.obj = None
        self.ro_model.min(self.ro_model.rc_model.vars[1][0].to_affine())
        sign = self.sign
        constr = (self.dec_vars[0] >= self.obj * sign)
        if isinstance(constr, ExpPWConstr):
            ro_constr_list = self.dro_to_roc(constr)
        elif isinstance(constr, PWConstr):
            ro_constr_list = []
            for piece in constr.pieces:
                ro_constr_list.extend(self.ro_to_roc(piece))
        elif isinstance(constr, DecCvxConstr):
            ro_constr_list = self.ro_to_roc(constr)
        elif constr.ctype == 'R':
            ro_constr_list = self.ro_to_roc(constr)
        elif constr.ctype == 'E':
            ro_constr_list = self.dro_to_roc(constr)
        else:
            raise TypeError('Unsupported objective function.')

        self.ro_model.st(ro_constr_list)

        # Event-wise Constraints
        for constr in self.all_constr:
            if isinstance(constr, ExpPWConstr):
                ro_constr_list = self.dro_to_roc(constr)
            elif isinstance(constr, (DecCvxConstr, DecPCvxConstr, DecExpConstr)):
                ro_constr_list = self.ro_to_roc(constr)
            elif constr.ctype == 'R':
                ro_constr_list = self.ro_to_roc(constr)
            elif constr.ctype == 'E':
                ro_constr_list = self.dro_to_roc(constr)
            else:
                raise ValueError('Unknown constraints.')
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
        # for event in constr.event_adapt:
        for s in range(self.num_scen):
            drule = drule_list[s]
            if isinstance(constr, DecRoConstr):
                is_equal = (all(constr.sense) if
                            isinstance(constr.sense, Iterable) else
                            constr.sense == 1)
                if is_equal:
                    roaffine = RoAffine(constr.raffine, constr.affine,
                                        constr.rand_model)
                    left = DecRoConstr(roaffine, 0,
                                       constr.event_adapt, constr.ctype)
                    left.ambset = constr.ambset
                    right = DecRoConstr(-roaffine, 0,
                                        constr.event_adapt, constr.ctype)
                    right.ambset = constr.ambset

                    return self.ro_to_roc(left) + self.ro_to_roc(right)

                raf_linear, aff_linear = (constr.raffine.linear[:, :num_var],
                                          constr.affine.linear[:, :num_var])

                row_ind = np.unique(raf_linear.indices)
                if isinstance(drule, RoAffine):
                    if len(row_ind) > 0:
                        if (drule.raffine[row_ind].linear.nnz > 0 or
                                np.any(drule.raffine[row_ind].const)):
                            raise SyntaxError('Incorrect affine expressions.')

                    raffine = raf_linear @ drule.affine
                    raffine = (raffine.reshape(constr.raffine.shape) +
                               constr.raffine.const)
                    roaffine = RoAffine(raffine, np.zeros(aff_linear.shape[0]),
                                        self.sup_model)
                    this_const = constr.affine.const
                    this_const = this_const.flatten()
                    roaffine = roaffine + aff_linear@drule + this_const
                    ew_constr = RoConstr(roaffine, constr.sense)

                elif isinstance(drule, Affine):
                    raffine = raf_linear @ drule
                    raffine = (raffine.reshape(constr.raffine.shape) +
                               constr.raffine.const)
                    this_const = constr.affine.const
                    this_const = this_const.flatten()
                    roaffine = RoAffine(raffine, aff_linear@drule + this_const,
                                        self.sup_model)
                    ew_constr = RoConstr(roaffine, constr.sense)
                else:
                    raise TypeError('Unknown type.')

            elif isinstance(constr, DecLinConstr):
                linear = constr.linear
                const = constr.const
                roaffine = linear @ drule - const.reshape(const.size)
                if isinstance(roaffine, RoAffine):
                    is_equal = (all(constr.sense) if
                                isinstance(constr.sense, Iterable) else
                                constr.sense == 1)
                    if is_equal == 1:

                        left = DecLinConstr(constr.model,
                                            constr.linear, constr.const,
                                            np.zeros(constr.linear.shape[0]),
                                            constr.event_adapt, constr.ctype)
                        right = DecLinConstr(constr.model,
                                             -constr.linear, -constr.const,
                                             np.zeros(constr.linear.shape[0]),
                                             constr.event_adapt, constr.ctype)
                        return self.ro_to_roc(left) + self.ro_to_roc(right)

                    left_empty = roaffine.raffine.linear.nnz == 0
                    right_empty = not roaffine.raffine.const.any()
                    if left_empty and right_empty:
                        ew_constr = LinConstr(roaffine.dec_model,
                                              roaffine.affine.linear,
                                              - roaffine.affine.const,
                                              constr.sense)
                    else:
                        ew_constr = RoConstr(roaffine, constr.sense)
                elif isinstance(roaffine, Affine):
                    ew_constr = LinConstr(roaffine.model, roaffine.linear,
                                          -roaffine.const, constr.sense)
                else:
                    raise TypeError('Unknown type.')
            elif isinstance(constr, DecPCvxConstr):
                linear_in = constr.affine_in.linear
                const_in = constr.affine_in.const
                aff_in = linear_in@drule + const_in.reshape(const_in.size)
                aff_in = aff_in.reshape(constr.affine_in.shape)

                if isinstance(constr.affine_scale, (Real, np.ndarray)):
                    aff_scale = constr.affine_scale
                else:
                    scale = constr.affine_scale.to_affine()
                    linear_sc = scale.linear
                    const_sc = scale.const
                    aff_scale = linear_sc@drule + const_sc.reshape(const_sc.size)
                aff_scale = aff_scale.reshape(constr.affine_scale.shape)

                if isinstance(aff_in, RoAffine):
                    aff_in = aff_in.affine
                if isinstance(constr.affine_out, (np.ndarray, Real)):
                    linear_out = np.zeros((constr.affine_out.size, drule.shape[0]))
                    const_out = constr.affine_out
                else:
                    linear_out = constr.affine_out.linear
                    const_out = constr.affine_out.const
                aff_out = linear_out@drule + const_out.reshape(const_out.size)
                if isinstance(aff_out, RoAffine):
                    aff_out = aff_out.affine
                aff_out = aff_out.reshape(constr.affine_out.shape)

                ew_constr = PCvxConstr(aff_in.model, aff_in, aff_scale, aff_out,
                                       constr.multiplier, constr.xtype)
            elif isinstance(constr, DecCvxConstr):
                linear_in = constr.affine_in.linear
                const_in = constr.affine_in.const
                aff_in = linear_in@drule + const_in.reshape(const_in.size)
                if isinstance(aff_in, RoAffine):
                    aff_in = aff_in.affine
                if isinstance(constr.affine_out, (np.ndarray, Real)):
                    linear_out = np.zeros((constr.affine_out.size, drule.shape[0]))
                    const_out = constr.affine_out
                else:
                    linear_out = constr.affine_out.linear
                    const_out = constr.affine_out.const
                aff_out = linear_out@drule + const_out.reshape(const_out.size)
                if isinstance(aff_out, RoAffine):
                    aff_out = aff_out.affine
                ew_constr = CvxConstr(aff_in.model, aff_in, aff_out,
                                      constr.multiplier, constr.xtype)
            elif isinstance(constr, DecExpConstr):
                if isinstance(drule, RoAffine):
                    drule_affine = drule.affine
                else:
                    drule_affine = drule

                affine1 = constr.expr1.to_affine()
                expr1 = affine1.linear@drule_affine + affine1.const

                if isinstance(constr.expr2, Real):
                    expr2 = constr.expr2
                else:
                    affine2 = constr.expr2.to_affine()
                    linear2 = affine2.linear
                    const2 = affine2.const
                    expr2 = linear2@drule_affine + const2

                if isinstance(constr.expr3, Real):
                    expr3 = constr.expr3
                else:
                    affine3 = constr.expr3.to_affine()
                    linear3 = affine3.linear
                    const3 = affine3.const
                    expr3 = linear3@drule_affine + const3

                ew_constr = ExpConstr(expr1.model, expr1, expr2, expr3)

            elif isinstance(constr, DecLMIConstr):
                if isinstance(drule, RoAffine):
                    drule_affine = drule.affine
                else:
                    drule_affine = drule

                lmi_left = constr.linear @ drule_affine - constr.const.flatten()
                lmi_linear = lmi_left.linear
                lmi_const = (-lmi_left.const).reshape((constr.dim, constr.dim))

                ew_constr = LMIConstr(lmi_left.model, lmi_linear, lmi_const, constr.dim)

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
                            support = ambset.sup_constr[s]
                    else:
                        ambset = constr.ambset
                        if isinstance(ambset, Ambiguity):
                            support = constr.ambset.sup_constr[s]
                        elif isinstance(ambset, Iterable):
                            support = ambset
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

        if isinstance(constr, ExpPWConstr):
            linears = []
            consts = []
            raffines = []
            for piece in constr.pieces:
                if isinstance(piece, DecLinConstr):
                    linears.append(piece.linear)
                    const = - piece.const
                    consts.append(const.reshape((const.size, 1)))
                    raffines.append(None)
                else:
                    linears.append(piece.affine.linear)
                    const = piece.affine.const
                    consts.append(const.reshape((const.size, 1)))
                    raffines.append(piece.raffine)
        elif isinstance(constr, DecLinConstr):
            linears = [constr.linear]
            const = - constr.const
            consts = [const.reshape((const.size, 1))]
            raffines = [None]
        else:
            linears = [constr.affine.linear]
            const = constr.affine.const
            consts = [const.reshape([const.size, 1])]
            raffines = [constr.raffine]

        ro_constr = []
        for i in range(linears[0].shape[0]):
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
                for linear, raffine, const in zip(linears, raffines, consts):
                    left = linear[i, :num_var] @ drule + const[i]
                    if raffine is not None:
                        if isinstance(drule, RoAffine):
                            extra = left.raffine
                            temp = drule.affine
                            left = left.affine.reshape(left.shape)
                        elif isinstance(drule, Affine):
                            extra = 0
                            temp = drule
                        else:
                            raise TypeError('Incorrect data type.')
                        row_ind = i*num_rand + np.arange(num_rand, dtype=int)
                        new_raffine = raffine.linear[row_ind] @ temp
                        new_raffine = new_raffine.reshape((1, new_raffine.size))
                        new_raffine += raffine.const[i, :num_rand] + extra
                        left = RoAffine(new_raffine, left, self.sup_model)

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

    def solve(self, solver=None, display=True, log=False, params={}):
        """
        Solve the model with the selected solver interface.

        Parameters
        ----------
            solver : {None, lpg_solver, clp_solver, ort_solver, eco_solver
                      cpx_solver, grb_solver, msk_solver, cpt_solver}
                Solver interface used for model solution. Use default solver
                if solver=None.
            display : bool
                True for displaying the solution information. False for hiding
                the solution information.
            log : bool
                True for printing the log information. False for hiding the log
                information. So far the argument only applies to Gurobi, CPLEX,
                and Mosek.
            params : dict
                A dictionary that specifies parameters of the selected solver.
                So far the argument only applies to Gurobi, CPLEX, and Mosek.
        """

        if solver is None:
            solution = def_sol(self.do_math(), display, log, params)
        else:
            solution = solver.solve(self.do_math(), display, log, params)

        if isinstance(solution, Solution):
            self.ro_model.solution = solution
        else:
            self.ro_model.solution = None

        self.ro_model.rc_model.solution = solution
        self.solution = solution

    def soc_solve(self, solver=None, degree=4, cuts=(-30, 60),
                  display=True, log=False, params={}):
        """
        Solve the model with the selected solver interface.

        Parameters
        ----------
            solver : {None, lpg_solver, clp_solver, ort_solver, eco_solver
                      cpx_solver, grb_solver, msk_solver, cpt_solver}
                Solver interface used for model solution. Use default solver
                if solver=None.
            degree : int
                The L-degree value for approximating exponential cone
                constraints.
            cuts : tuple of two integers
                The lower and upper cut-off values for the SOC approximation
                of exponential constraints.
            display : bool
                True for displaying the solution information. False for hiding
                the solution information.
            log : bool
                True for printing the log information. False for hiding the log
                information. So far the argument only applies to Gurobi, CPLEX,
                and Mosek.
            params : dict
                A dictionary that specifies parameters of the selected solver.
                So far the argument only applies to Gurobi, CPLEX, and Mosek.
        """

        formula = self.do_math().to_socp(degree, cuts)
        if solver is None:
            solution = def_sol(formula, display, log, params)
        else:
            solution = solver.solve(formula, display, log, params)

        if isinstance(solution, Solution):
            self.ro_model.solution = solution
        else:
            self.ro_model.solution = None

        self.ro_model.rc_model.solution = solution
        self.solution = solution

    def get(self):
        """
        Return the optimal objective value of the solved model.

        Notes
        -----
        An error message is given if the model is unsolved or no solution
        is obtained.
        """

        if self.solution is None:
            raise RuntimeError('The model is unsolved or no solution is obtained')

        solution = self.solution
        if np.isnan(solution.objval):
            msg = 'No solution available. '
            msg += f'{solution.solver} solution status: {solution.status}'
            raise RuntimeError(msg)

        return self.sign * self.solution.objval

    def optimal(self):

        if self.solution is None:
            return False
        else:
            return not np.isnan(self.solution.objval)


class Ambiguity:
    """
    The Ambiguity class creates an ambiguity set object
    """

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
        """
        Return a data frame showing how the event-wise support sets and
        uncertainty sets of random variable expectations are defined.
        """

        num_scen = self.model.num_scen
        table = pd.DataFrame([False]*num_scen,
                             columns=['support'],
                             index=self.s.series.index)
        sup_constr = self.sup_constr
        defined = pd.notnull(pd.Series(sup_constr,
                                       index=self.s.series.index))
        table.loc[defined, 'support'] = True

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
        """
        Specify the support set(s) of an ambiguity set.

        Parameters
        ----------
        args : Constraints or iterables
            Constraints or collections of constraints as iterable type of
            objects, used for defining the feasible region of the support set.

        Notes
        -----
        RSOME leaves the support set unspecified if the given argument is
        an empty iterable object.
        """

        self.update = True
        return self.s.suppset(*args)

    def exptset(self, *args):
        """
        Specify the uncertainty set of the expected values of random
        variables for an ambiguity set.

        Parameters
        ----------
        args : tuple
            Constraints or collections of constraints as iterable type of
            objects, used for defining the feasible region of the uncertainty
            set of expectations.

        Notes
        -----
        RSOME leaves the uncertainty set of expectations unspecified if the
        input argument is an empty iterable object.
        """

        self.update = True
        return self.s.exptset(*args)

    def probset(self, *args):
        """
        Specify the uncertainty set of the scenario probabilities for an
        ambiguity set.

        Parameters
        ----------
        args : tuple
            Constraints or collections of constraints as iterable type of
            objects, used for defining the feasible region of the uncertainty
            set of scenario probabilities.

        Notes
        -----
        RSOME leaves the uncertainty set of probabilities unspecified if the
        input argument is an empty iterable object.
        """

        self.update = True
        for arg in args:
            if arg.model is not self.model.pro_model:
                raise ValueError('Constraints are not defined for the ' +
                                 'probability set.')

        pr = self.model.p
        self.pro_constr = [pr >= 0, pr.sum() == 1] + list(args)

    def mix_support(self, primal=True):

        if not self.update and self.mix_model is not None:
            return self.mix_model.do_math(primal, obj=False)

        self.model.pro_model.reset()
        self.model.pro_model.st(self.pro_constr)
        pro_support = self.model.pro_model.do_math(obj=False)
        self.mix_model = GCPModel(nobj=True, mtype='M', top=self.model)

        # Constraints for probabilities
        p = self.mix_model.dvar(pro_support.linear.shape[1])
        constr = LinConstr(self.mix_model,
                           pro_support.linear, pro_support.const,
                           pro_support.sense)
        self.mix_model.st(constr)
        for q in pro_support.qmat:
            cconstr = ConeConstr(self.mix_model, p, q[1:], p, q[0])
            self.mix_model.st(cconstr)
        for ex in pro_support.xmat:
            econstr = ExpConstr(self.mix_model, p[ex[0]], p[ex[1]], p[ex[2]])
            self.mix_model.st(econstr)

        # Constraints for expectations
        for econstr, indices in zip(self.exp_constr, self.exp_constr_indices):
            self.model.exp_model.reset()
            self.model.exp_model.st(econstr)
            exp_support = self.model.exp_model.do_math(obj=False)
            exp_var = self.mix_model.dvar(exp_support.linear.shape[1])
            affine = (exp_support.linear @ exp_var -
                      p[indices].sum() * exp_support.const)
            constr = LinConstr(affine.model, affine.linear, affine.const,
                               exp_support.sense)
            self.mix_model.st(constr)
            for q in exp_support.qmat:
                cconstr = ConeConstr(self.mix_model, exp_var, q[1:],
                                     exp_var, q[0])
                self.mix_model.st(cconstr)

        return self.mix_model.do_math(primal, obj=False)

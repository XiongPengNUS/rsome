from .socp import Model as SOCModel
from .ro import Model as ROModel
# from .lp import LinConstr, Bounds, CvxConstr, ConeConstr
from .lp import Vars, VarSub    # , Affine, Convex
# from .lp import RoAffine, RoConstr
from .subroutines import *
import numpy as np
import pandas as pd
# import scipy.sparse as sp
# from numbers import Real
# from scipy.sparse import csr_matrix
from collections import Iterable, Sized


class Model:

    def __init__(self, name=None):

        self.ro_model = ROModel()
        self.vt_model = SOCModel(mtype='V')
        self.sup_model = SOCModel(nobj=True, mtype='S')
        self.exp_model = SOCModel(nobj=True, mtype='E')
        # self.pro_model = SOCModel(nobj=True, mtype='P')
        self.pro_model = None

        self.all_constr = []

        self.num_scen = 1
        self.sup_constr = []
        self.exp_constr = []
        self.exp_constr_indices = []
        self.pro_constr = []
        self.pr = None

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
        return RandVar(sup_var, exp_var)

    def dvar(self, shape=(1,), vtype='C', name=None):

        dec_var = self.vt_model.dvar(shape, vtype, name)
        return DecVar(self, dec_var, name)

    def ambiguity(self, scens=1):

        if self.all_constr:
            raise SyntaxError('Ambiguity set must be specified ' +
                              'before defining constraints.')

        if isinstance(scens, int):
            num_scen = scens
            series = pd.Series(range(num_scen), dtype=np.int32)
        elif isinstance(scens, Sized):
            num_scen = len(scens)
            series = pd.Series(range(num_scen), index=scens, dtype=np.int32)
        else:            raise ValueError('Incorrect scenarios.')

        return Ambiguity(self, num_scen, series)

    def bou(self, *args):

        bou_set = self.ambiguity()
        for arg in args:
            if arg.model is not self.sup_model:
                raise ValueError('Constraints are not for this support.')

        self.sup_constr[0] = tuple(args)

        return bou_set

    def wks(self, *args):

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

        self.sup_constr = [tuple(sup_constr)]
        self.exp_constr = [tuple(exp_constr)]
        self.exp_constr_indices = [np.array([0], dtype=np.int32)]

        return wks_set


class DecVar(Vars):

    def __init__(self, model, dvars, name=None):

        super().__init__(dvars.model, dvars.first, dvars.shape,
                         dvars.vtype, dvars.name)
        self.rsome_model = model
        self.event_adapt = None
        self.rand_adapt = None
        self.name = name

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).reshape((1, ) * self.ndim)

        return DecVarSub(self.rsome_model, self, indices)

    def adapt(self, to):

        if isinstance(to, Scen):
            self.evtadapt(to)
        elif isinstance(to, (RandVar, RandVarSub)):
            self.affadapt(to)
        else:
            raise ValueError('Can not define adaption for the inputs.')

    def evtadapt(self, scens):

        if self.event_adapt is None:
            self.event_adapt = [list(range(self.rsome_model.num_scen))]

        events = scens.series
        events = list(events) if isinstance(events, Iterable) else [events]

        for event in events:
            if event in self.event_adapt[0]:
                self.event_adapt[0].remove(event)
            else:
                raise ValueError('The scenario indexed by {0} '.foramt(event) +
                                 'is redefined.')

        if not self.event_adapt[0]:
            self.event_adapt.pop(0)

        self.event_adapt.append(events)

    def affadapt(self, rvars):

        self[:].affadapt(rvars)


class DecVarSub(VarSub):

    def __init__(self, model, dvars, indices):

        super().__init__(dvars, indices)
        self.rsome_model = model
        self.event_adapt = dvars.event_adapt
        self.rand_adapt = dvars.rand_adapt
        self.dvars = dvars

    def adapt(self, rvars):

        if not isinstance(rvars, (RandVar, RandVarSub)):
            raise TypeError('Affine adaptation requires a random variable.')

        self.affadapt(rvars)

    def affadapt(self, rvars):

        if self.rand_adapt is None:
            sup_model = self.rsome_model.sup_model
            self.rand_adapt = np.zeros((self.size, sup_model.vars[-1].last),
                                       dtype=np.int8)

        dec_indices = self.indices
        dec_indices = dec_indices.reshape((dec_indices.size, 1))
        rand_indices = rvars.get_ind()

        if self.rand_adapt[dec_indices, rand_indices].any():
            raise SyntaxError('Redefinition of adaptation is not allowed.')

        self.rand_adapt[dec_indices, rand_indices] = 1
        print(self.rand_adapt)
        self.dvars.rand_adapt = self.rand_adapt


class RandVar(Vars):

    def __init__(self, svars, evars):

        super().__init__(svars.model, svars.first,
                         svars.shape, svars.vtype, svars.name, svars.sparray)
        self.e = evars

    @property
    def E(self):

        return self.e

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).reshape((1, ) * self.ndim)

        return RandVarSub(self, indices)


class RandVarSub(VarSub):

    def __init__(self, rvars, indices):

        super().__init__(rvars, indices)
        self.e = VarSub(rvars.e, indices)

    @property
    def E(self):

        return self.e


class Ambiguity:

    def __init__(self, model, num_scen, scens):

        model.num_scen = num_scen
        self.model = model

        model.sup_constr = [None] * num_scen
        model.exp_constr = []
        model.pro_model = SOCModel(nobj=True, mtype='P')
        pr = model.pro_model.dvar(num_scen, name='probabilities')
        self.p = pr
        model.pro_constr = [pr >= 0, pr.sum() == 1]

        self.s = Scen(self, scens, self.p)

    def __getitem__(self, indices):

        return self.s[indices]

    def showevents(self):

        num_scen = self.model.num_scen
        table = pd.DataFrame(['undefined']*num_scen,
                             columns=['support'], index=self.s.series.index)
        sup_constr = self.model.sup_constr
        if sup_constr is not None:
            defined = pd.notnull(pd.Series(sup_constr,
                                 index=self.s.series.index))
            table.loc[defined, 'support'] = 'defined'

        # exp_cosntr = self.model.exp_constr
        exp_constr_indices = self.model.exp_constr_indices
        count = 0
        if exp_constr_indices is not None:
            for indices in exp_constr_indices:
                column = 'expectation {0}'.format(count)
                count += 1
                table[column] = False
                table.iloc[indices, count] = True

        return table

    def scenarios(self):

        return self.s, self.p

    @property
    def loc(self):

        return self.s.loc

    @property
    def iloc(self):

        return self.s.iloc

    def suppset(self, *args):

        return self.s.suppset(*args)

    def exptset(self, *args):

        return self.s.exptset(*args)

    def probset(self, *args):

        for arg in args:
            if arg.model is not self.model.pro_model:
                raise ValueError('Constraints are not defined for the ' +
                                 'probability set.')

        pr = self.p
        self.model.pro_constr = [pr >= 0, pr.sum() == 1] + list(args)


class Scen:

    def __init__(self, ambset, series, pr):

        # super().__init__(data=series.values, index=series.index)
        self.ambset = ambset
        self.series = series
        self.p = pr

    def __str__(self):

        return 'Scenario indices: \n' + self.series.__str__()

    def __repr__(self):

        return self.__str__()

    def __getitem__(self, indices):

        indices_p = self.series[indices]
        return Scen(self.ambset, self.series[indices], self.p[indices_p])

    @property
    def loc(self):

        return ScenLoc(self)

    @property
    def iloc(self):

        return ScenILoc(self)

    def suppset(self, *args):

        for arg in args:
            if arg.model is not self.ambset.model.sup_model:
                raise ValueError('Constraints are not for this support.')

        # for i in self.series:
        indices = (self.series if isinstance(self.series, pd.Series)
                   else [self.series])
        for i in indices:
            self.ambset.model.sup_constr[i] = tuple(args)

    def exptset(self, *args):

        for arg in args:
            if arg.model is not self.ambset.model.exp_model:
                raise ValueError('Constraints are not defined for ' +
                                 'expectation sets.')

        self.ambset.model.exp_constr.append(tuple(args))
        self.ambset.model.exp_constr_indices.append(self.series)


class ScenLoc:

    def __init__(self, scens):

        self.scens = scens
        self.indices = []

    def __getitem__(self, indices):

        indices_s = self.scens.series.loc[indices]

        return Scen(self.scens.ambset, indices_s, self.scens.p[indices_s])


class ScenILoc:

    def __init__(self, scens):

        self.scens = scens
        self.indices = []

    def __getitem__(self, indices):

        indices_s = self.scens.series.iloc[indices]

        return Scen(self.scens.ambset, indices_s, self.scens.p[indices_s])

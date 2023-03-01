from .gcp import Model as GCPModel
from .gcp import GCProg
from .lp import LinConstr, Bounds, CvxConstr, ExpConstr
from .lp import Vars, VarSub, Affine, Convex
from .lp import def_sol
from .socp import Model as SOCModel
from .socp import SOCProg
from collections.abc import Iterable
from numbers import Real
from scipy.sparse import csr_matrix

import numpy as np
import scipy.sparse as sp


class Model:

    def __init__(self, name=None):

        self.master = GCPModel(nobj=False, mtype='R', name=name, top=self)
        self.here_now_index = []
        self.wait_see_index = []
        self.all_here_now_index = []
        self.rand_index = []

        self.subs = []

        self.exact_bounds = []
        self.relaxed_bounds = []
        self.solutions = []

    def dvar(self, shape=(), vtype='C', name=None):

        new_var = self.master.dvar(shape, vtype, name, aux=False)
        index = range(new_var.first, new_var.last)
        self.here_now_index.extend(index)

        return HaNVar(new_var)

    def rvar(self, shape=(), name=None):

        new_var = self.master.dvar(shape, 'C', name, aux=False)
        index = range(new_var.first, new_var.last)
        self.rand_index.extend(index)

        return RandVar(new_var)

    def min(self, obj):

        if isinstance(obj, (Vars, VarSub, Affine)):
            affine = obj.to_affine()
            if affine.linear[:, self.wait_see_index].nnz > 0:
                msg = 'No wait-and-see decision is allowed in master problem objective.'
                raise ValueError(msg)
            if affine.linear[:, self.rand_index].nnz > 0:
                msg = 'No random variable is allowed in master problem objective.'
                raise ValueError(msg)
        elif isinstance(obj, Convex):
            cond1 = obj.affine_in.linear[:, self.wait_see_index].nnz > 0
            cond2 = obj.affine_out.linear[:, self.wait_see_index].nnz > 0
            if cond1 or cond2:
                msg = 'No wait-and-see decision is allowed in master problem objective.'
                raise ValueError(msg)
            cond1 = obj.affine_in.linear[:, self.rand_index].nnz > 0
            cond2 = obj.affine_out.linear[:, self.rand_index].nnz > 0
            if cond1 or cond2:
                msg = 'No random variable is allowed in master problem objective.'
                raise ValueError(msg)
        else:
            raise TypeError('Unsupported objective function.')

        self.master.min(obj)

    def max(self, obj):

        if isinstance(obj, (Vars, VarSub, Affine)):
            affine = obj.to_affine()
            if affine.linear[:, self.wait_see_index].nnz > 0:
                msg = 'No wait-and-see decision is allowed in master problem objective.'
                raise ValueError(msg)
            if affine.linear[:, self.rand_index].nnz > 0:
                msg = 'No random variable is allowed in master problem objective.'
                raise ValueError(msg)
        elif isinstance(obj, Convex):
            cond1 = obj.affine_in.linear[:, self.wait_see_index].nnz > 0
            cond2 = obj.affine_out.linear[:, self.wait_see_index].nnz > 0
            if cond1 or cond2:
                msg = 'No wait-and-see decision is allowed in master problem objective.'
                raise ValueError(msg)
            cond1 = obj.affine_in.linear[:, self.rand_index].nnz > 0
            cond2 = obj.affine_out.linear[:, self.rand_index].nnz > 0
            if cond1 or cond2:
                msg = 'No random variable is allowed in master problem objective.'
                raise ValueError(msg)
        else:
            raise TypeError('Unsupported objective function.')

        self.master.max(obj)

    def st(self, *arg):

        for constr in arg:
            if isinstance(constr, Iterable):
                for item in constr:
                    self.st(item)
            else:
                if isinstance(constr, LinConstr):
                    if constr.linear[:, self.wait_see_index].nnz > 0:
                        msg = 'No wait-and-see decision is allowed in master problem.'
                        raise ValueError(msg)
                    if constr.linear[:, self.rand_index].nnz > 0:
                        msg = 'No random variable is allowed in master problem.'
                        raise ValueError(msg)

                elif isinstance(constr, CvxConstr):
                    cond1 = constr.affine_in.linear[:, self.wait_see_index].nnz > 0
                    cond2 = constr.affine_out.linear[:, self.wait_see_index].nnz > 0
                    if cond1 or cond2:
                        msg = 'No wait-and-see decision is allowed in master problem.'
                        raise ValueError(msg)
                    cond1 = constr.affine_in.linear[:, self.rand_index].nnz > 0
                    cond2 = constr.affine_out.linear[:, self.rand_index].nnz > 0
                    if cond1 or cond2:
                        msg = 'No random variable is allowed in master problem.'
                        raise ValueError(msg)

                elif isinstance(constr, Bounds):
                    if set(constr.indices).intersection(self.wait_see_index):
                        msg = 'No wait-and-see decision is allowed in master problem.'
                        raise ValueError(msg)
                    if set(constr.indices).intersection(self.rand_index):
                        msg = 'No random variable is allowed in master problem.'
                        raise ValueError(msg)

                elif isinstance(constr, ExpConstr):
                    print('TODO!')
                    raise ValueError('Not yet.')

                else:
                    raise TypeError('Unsupported constraints.')

                self.master.st(constr)

    def subprob(self, size=1):

        sub_models = SubModel(self, size)
        new_var = self.master.dvar(size)
        sub_models.values = new_var
        sub_models.value_index = list(range(new_var.first, new_var.last))

        self.subs.append(sub_models)

        return sub_models

    def do_math(self):

        formula = self.master.do_math()
        all_indices = np.arange(0, formula.linear.shape[1])
        indices = np.setxor1d(all_indices,
                              np.array(self.wait_see_index + self.rand_index),
                              assume_unique=True)

        formula = GCProg(formula.linear[:, indices], formula.const, formula.sense,
                         formula.vtype[indices],
                         formula.ub[indices], formula.lb[indices],
                         formula.qmat, formula.xmat, formula.obj[:, indices])

        self.all_here_now_index = indices

        return formula

    def solve(self, solver=None, display=True, params={}):

        self.master.solve(solver, display, params)
        # self.relaxed_bounds.append(self.master.get())
        self.solutions.append(self.master.solution)

    def add_cut(self, cuts):

        add_linear = sp.vstack([cut.linear for cut in cuts])
        add_const = np.concatenate([cut.const for cut in cuts])
        add_sense = np.zeros(len(cuts))

        formula = self.master.primal
        formula.linear = sp.vstack((formula.linear, add_linear), format='csr')
        formula.const = np.concatenate((formula.const, add_const))
        formula.sense = np.concatenate((formula.sense, add_sense))
        self.master.primal = formula

    def update(self, solver=None, master_display=True, sub_display=False,
               master_params={}, sub_params={}):

        self.solve(solver, master_display, master_params)
        self.relaxed_bounds.append(self.master.get())
        obj_linear = self.master.obj.linear.toarray()[0]
        obj_const = self.master.obj.const
        indices = self.here_now_index
        x_current = self.master.solution.x[indices]
        exact_obj = obj_linear[self.here_now_index] @ x_current + obj_const

        for sub in self.subs:
            sol = sub.solve(solver, sub_display, sub_params)
            objvals = sol['objval']
            exact_obj += obj_linear[sub.value_index] @ np.array(objvals)

            self.add_cut(sol['cut'])

        self.exact_bounds.append(exact_obj.reshape(()))

    def get_bounds(self):

        if self.master.sign > 0:
            lower = max(self.relaxed_bounds)
            upper = min(self.exact_bounds)
        else:
            lower = max(self.exact_bounds)
            upper = min(self.relaxed_bounds)

        return lower, upper

    def get_gap(self, relative=True):

        lower, upper = self.get_bounds()

        if relative:
            return (upper - lower) / lower
        else:
            return upper - lower

    def get(self):

        if self.exact_bounds == []:
            raise RuntimeError('The model is unsolved or no solution is obtained.')
        else:
            if self.master.sign > 0:
                return min(self.exact_bounds)
            else:
                return max(self.exact_bounds)


class SubModel:

    def __init__(self, model, size):

        self.model = model
        self.obj = None
        self.sign = None
        self.size = size
        self.lin_constr = []
        self.bounds = []
        self.values = None
        self.value_index = None
        self.wait_see_index = []
        self.uset = [None] * size

        self.formula = None
        self.update = True

    def __getitem__(self, item):

        return SubModelIter(self, item)

    def def_obj(self, obj, sign):

        if isinstance(obj, (Vars, VarSub, Affine)):
            affine = obj.to_affine()
            if affine.linear[:, self.model.here_now_index].nnz > 0:
                msg = 'Here-and-now decisions are not allowed in sub-problem objective.'
                raise ValueError(msg)
            if affine.linear[:, self.model.rand_index].nnz > 0:
                msg = 'Random variables are not allowed in sub-problem objective.'
                raise ValueError(msg)
            value_index = list(range(self.values.first, self.values.last))
            if affine.linear[:, value_index].nnz > 0:
                msg = 'Sub-problem values are not allowed in sub-problem objective.'
                raise ValueError(msg)
        else:
            raise TypeError('Unsupported objective function')

        self.obj = obj
        self.sign = sign

    def dvar(self, shape=(), name=None):

        new_var = self.model.master.dvar(shape, 'C', name, aux=False)
        index = range(new_var.first, new_var.last)
        self.model.wait_see_index.extend(index)
        self.wait_see_index.extend(index)

        self.update = True

        return new_var

    def min(self, obj):

        self.def_obj(obj, 1)

        self.update = True

    def max(self, obj):

        self.def_obj(obj, -1)

        self.update = True

    def st(self, *arg):

        for constr in arg:
            if isinstance(constr, Iterable):
                for item in constr:
                    self.st(item)
            else:
                value_index = list(range(self.values.first, self.values.last))
                if isinstance(constr, LinConstr):
                    if constr.linear[:, value_index].nnz > 0:
                        msg = 'Sub-problem values are not allowed in sub-problem.'
                        raise ValueError(msg)
                    if constr.linear[:, self.model.wait_see_index].nnz == 0:
                        msg = 'No wait-and-see decision appears in the constraint.'
                        raise ValueError(msg)
                    self.lin_constr.append(constr)
                elif isinstance(constr, Bounds):
                    if not set(constr.indices).intersection(self.model.wait_see_index):
                        msg = 'No wait-and-see decision appears in the constraint.'
                        raise ValueError(msg)
                    self.bounds.append(constr)
                else:
                    raise TypeError('Unsupported constraints.')

        self.update = True

    def uncertain(self, uncertainty):

        self.uset = [uncertainty] * self.size

    def do_math(self):

        if self.update is False and self.formula is not None:
            return self.formula

        sub_model = SOCModel(nobj=True, mtype='R')

        vars = []
        for var in self.model.master.vars:
            if var.vtype == 'C':
                vars.append(var)
            else:
                new_var = Vars(var.model, var.first, var.shape, 'C',
                               var.name, var.sparray)
                vars.append(new_var)
        sub_model.vars = vars + self.model.master.auxs
        sub_model.lin_constr = self.lin_constr
        sub_model.bounds = self.bounds

        dual = sub_model.do_math(primal=False, obj=False)
        dual.const = self.obj.linear.toarray()[0] * dual.const * self.sign

        vtype = dual.vtype
        ub = dual.ub
        lb = dual.lb
        qmat = dual.qmat
        obj = dual.obj

        linear = dual.linear[self.wait_see_index, :]
        const = dual.const[self.wait_see_index]
        sense = dual.sense[self.wait_see_index]

        cmat = dual.linear[self.model.rand_index, :].T
        if self.model.all_here_now_index == []:
            all_indices = np.arange(0, dual.linear.shape[0])
            not_here_now = self.model.wait_see_index + self.model.rand_index
            indices = np.setxor1d(all_indices, np.array(not_here_now),
                                  assume_unique=True)
            self.model.all_here_now_index = indices
        else:
            indices = self.model.all_here_now_index

        amat = dual.linear[indices, :].T

        self.update = False

        return SubProg(linear, const, sense, vtype, ub, lb,
                       qmat, amat, cmat, obj)

    def solve(self, solver=None, display=False, params={}):

        formula = self.do_math()

        indices = self.model.all_here_now_index
        x_here_now = self.model.master.solution.x[indices]

        cuts = []
        pis = []
        objvals = []
        for i, uncertainty in enumerate(self.uset):
            z_values = uncertainty.center
            z_values = z_values.reshape((z_values.size, ))

            obj = formula.obj + formula.amat@x_here_now
            obj += formula.cmat @ z_values
            obj = obj.reshape((1, obj.size))

            sub_formula = SOCProg(formula.linear, formula.const, formula.sense,
                                  formula.vtype, formula.ub, formula.lb,
                                  formula.qmat, obj)

            if solver is None:
                sub_sol = def_sol(sub_formula, display, params)
            else:
                sub_sol = solver.solve(sub_formula, display, params)

            pi = sub_sol.x
            pis.append(pi)
            objvals.append(-sub_sol.objval * self.sign)

            cut_linear = - (pi @ formula.amat)
            cut_linear = csr_matrix(cut_linear.reshape((1, cut_linear.size)))
            cut_const = (formula.obj - formula.cmat@z_values) @ pi
            left = Affine(self.model.master, cut_linear, cut_const)
            right = self.values[i]
            cut = left <= self.sign * right
            cuts.append(cut)

        return {'pi': pis, 'objval': objvals, 'cut': cuts}


class SubModelIter:

    def __init__(self, submodel, item):

        self.submodel = submodel
        self.item = item

    def uncertain(self, uncertainty):

        if isinstance(uncertainty, Uncertainty):
            self.submodel.uset[self.item] = uncertainty
        else:
            raise TypeError('Unsupported uncertainty set.')


class SubProg(SOCProg):

    def __init__(self, linear, const, sense, vtype, ub, lb,
                 qmat, amat, cmat, obj):

        super().__init__(linear, const, sense, vtype, ub, lb, qmat, obj)
        self.amat = amat
        self.cmat = cmat


class RandVar(Vars):

    def __init__(self, vars):

        super().__init__(vars.model, vars.first, vars.shape, vars.vtype,
                         vars.name, vars.sparray)

    def __repr__(self):

        string = super().__repr__()

        return string.replace('continuous ', '').replace('decision', 'random')

    def singleton(self, center):

        if isinstance(center, Real):
            center = np.array([center])

        center = center.reshape(self.shape)

        return Uncertainty(self, center, 0, utype='S')

    def get(self):

        raise TypeError('Cannot access the solution of random variables')


class HaNVar(Vars):

    def __init__(self, vars):

        super().__init__(vars.model, vars.first, vars.shape, vars.vtype,
                         vars.name, vars.sparray)

    def get(self):

        top = self.model.top
        exact_bounds = np.array(top.exact_bounds)
        index = np.argmin(top.master.sign * exact_bounds)

        xvec = top.solutions[index].x[self.first:self.last]

        return xvec.reshape(self.shape)


class Uncertainty:

    def __init__(self, vars, center, radius, utype):

        self.vars = vars
        self.center = center
        self.radius = radius
        self.utype = utype

    def __repr__(self):

        utypes = {'S': 'singleton',
                  '1': 'one-norm',
                  '2': 'two-norm',
                  'I': 'infinite-norm'}

        article = 'an' if self.utype == 'I' else 'a'
        string = f'{article} {utypes[self.utype].lower()} uncertainty set '
        string += f'for {self.vars.__repr__()}'

        return string

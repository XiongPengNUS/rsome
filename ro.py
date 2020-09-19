from .socp import Model as SOCModel
from .lp import LinConstr, Bounds, CvxConstr
from .lp import Vars, VarSub, Affine, Convex
from .lp import RoAffine, RoConstr
from .subroutines import *
import numpy as np
from scipy.sparse import csr_matrix
from collections import Iterable


class Model:
    """
    The Model class creates an object of robust optimization models
    """

    def __init__(self, name=None):

        self.rc_model = SOCModel(mtype='R')
        self.sup_model = SOCModel(nobj=True, mtype='S')

        self.all_constr = []

        self.obj = None
        self.obj_support = None
        self.sign = 1

        self.primal = None
        self.dual = None
        self.solution = None
        self.pupdate = True
        self.dupdate = True

        self.solution = None

        self.name = name

    def reset(self):

        self.all_constr = []
        self.pupdate = True
        self.dupdate = True
        self.primal = None
        self.dual = None
        self.rc_model.reset()

    def dvar(self, shape=(1,), vtype='C', name=None, aux=False):
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
        aux : leave it unspecified.

        Returns
        -------
        new_var : rsome.lp.Vars
            An array of new decision variables
        """

        new_var = self.rc_model.dvar(shape, vtype, name, aux)
        return new_var

    def rvar(self, shape=(1,), name=None):

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
        new_var : rsome.lp.Vars
            An array of new random variables
        """

        new_var = self.sup_model.dvar(shape, 'C', name)
        return new_var

    def ldr(self, shape=(1,), name=None):

        """
        Returns an array with the given shape of linear decision rule
        variables.

        Parameters
        ----------
        shape : int or tuple
            Shape of the variable array.
        name : str
            Name of the variable array

        Returns
        -------
        new_var : rsome.ro.DecRule
            An array of new linear decision rule variables
        """

        new_ldr = DecRule(self, shape, name)
        return new_ldr

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

    def minmax(self, obj, *args):

        if np.prod(obj.shape) > 1:
            raise ValueError('Incorrect function dimension.')

        constraints = []
        for items in args:
            if isinstance(items, Iterable):
                constraints.extend(list(items))
            else:
                constraints.append(items)

        sup_model = self.sup_model
        sup_model.reset()
        for item in constraints:
            if item.model is not sup_model:
                raise SyntaxError('Models mismatch.')
            sup_model.st(item)

        self.obj = obj
        self.obj_support = sup_model.do_math(primal=False)
        self.sign = 1
        self.pupdate = True
        self.dupdate = True

    def maxmin(self, obj, *args):

        if np.prod(obj.shape) > 1:
            raise ValueError('Incorrect function dimension.')

        constraints = []
        for items in args:
            if isinstance(items, Iterable):
                constraints.extend(list(items))
            else:
                constraints.append(items)

        sup_model = self.sup_model
        sup_model.reset()
        for item in constraints:
            if item.model is not sup_model:
                raise SyntaxError('Models mismatch.')
            sup_model.st(item)

        self.obj = obj
        self.obj_support = sup_model.do_math(primal=False)
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def st(self, *arg):

        for constr in arg:
            if isinstance(constr, Iterable):
                for item in constr:
                    self.st(item)

            elif isinstance(constr, (LinConstr, Bounds, CvxConstr)):
                if (constr.model is not self.rc_model) or \
                        (constr.model.mtype != 'R'):
                    raise ValueError('Models mismatch.')
                self.all_constr.append(constr)
            elif isinstance(constr, RoConstr):
                if (constr.dec_model is not self.rc_model) or \
                        (constr.rand_model is not self.sup_model):
                    raise ValueError('Models mismatch.')
                sense = (constr.sense[0] if isinstance(constr.sense,
                                                       np.ndarray)
                         else constr.sense)
                if sense == 0:
                    self.all_constr.append(constr)
                else:
                    left = RoAffine(constr.raffine, constr.affine,
                                    constr.rand_model)
                    right = RoAffine(-constr.raffine, -constr.affine,
                                     constr.rand_model)
                    self.all_constr.append(RoConstr(left, sense=0))
                    self.all_constr.append(RoConstr(right, sense=0))
            else:
                raise TypeError('Unknown type of constraints')

    def do_math(self, primal=True):

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal
        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

        self.rc_model.reset()
        if isinstance(self.obj, (Vars, VarSub, Affine, Convex)):
            self.rc_model.obj = self.obj
            self.rc_model.sign = self.sign
            more_roc = []
        elif isinstance(self.obj, RoAffine):
            obj_constr = (self.rc_model.vars[0] >= self.sign * self.obj)
            obj_constr.support = self.obj_support
            more_roc = [obj_constr]
        else:
            raise TypeError('Incorrect type for the objective function.')

        for constr in self.all_constr + more_roc:
            if isinstance(constr, (LinConstr, Bounds, CvxConstr)):
                self.rc_model.st(constr)
            if isinstance(constr, RoConstr):
                if constr.support:
                    rc_constrs = constr.le_to_rc()
                else:
                    rc_constrs = constr.le_to_rc(self.obj_support)
                for rc_constr in rc_constrs:
                    self.rc_model.st(rc_constr)

        formula = self.rc_model.do_math(primal, obj=True)

        if primal:
            self.primal = formula
            self.pupdate = False
        else:
            self.dual = formula
            self.dupdate = False

        return formula

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

        self.rc_model.solution = solver.solve(self.do_math(), display, export)
        self.solution = self.rc_model.solution

    def get(self):

        if self.rc_model.solution is None:
            raise SyntaxError('The model is unsolved or no feasible solution.')
        return self.sign * self.rc_model.solution.objval


class DecRule:

    __array_priority__ = 102

    def __init__(self, model, shape=(1,), name=None,):

        self.model = model
        self.name = name
        self.fixed = model.dvar(shape, 'C')
        self.shape = self.fixed.shape
        self.size = np.prod(self.shape)
        self.depend = None
        self.roaffine = None
        self.var_coeff = None

    def __str__(self):

        suffix = 's' if np.prod(self.shape) > 1 else ''

        string = '' if self.name is None else self.name + ': '
        string += 'x'.join([str(size) for size in self.shape]) + ' '
        string += 'decision rule variable' + suffix

        return string

    def __repr__(self):

        return self.__str__()

    def reshape(self, shape):

        return self.to_affine().reshape(shape)

    def adapt(self, rvar, ldr_indices=None):

        if self.roaffine is not None:
            raise SyntaxError('Adaptation must be defined ' +
                              'before used in constraints')

        if self.depend is None:
            self.depend = np.zeros((self.size,
                                    self.model.sup_model.vars[-1].last),
                                   dtype=int)

        indices = rvar.get_ind()
        if ldr_indices is None:
            ldr_indices = np.arange(self.depend.shape[0], dtype=int)
        ldr_indices = ldr_indices.reshape((ldr_indices.size, 1))

        row_ind = (ldr_indices *
                   np.ones(indices.shape, dtype=int)).flatten()
        col_ind = (np.ones(ldr_indices.shape, dtype=int) * indices).flatten()

        if self.depend[row_ind, col_ind].any():
            raise SyntaxError('Redefinition of adaptation is not allowed.')

        self.depend[ldr_indices, indices] = 1

    def to_affine(self):

        if self.roaffine is not None:
            return self.roaffine
        else:
            if self.depend is not None:
                num_ones = self.depend.sum()
                var_coeff = self.model.dvar(num_ones)
                self.var_coeff = var_coeff
                row_ind = np.where(self.depend.flatten() == 1)[0]
                col_ind = var_coeff.get_ind()
                num_rand = self.model.sup_model.vars[-1].last
                row = self.size * num_rand
                col = self.model.rc_model.vars[-1].last
                raffine_linear = csr_matrix((np.ones(num_ones),
                                             (row_ind, col_ind)),
                                            shape=(row, col))
                raffine = Affine(self.model.rc_model,
                                 raffine_linear,
                                 np.zeros((self.size, num_rand)))
                roaffine = RoAffine(raffine, np.zeros(self.shape),
                                    self.model.sup_model)
                self.roaffine = self.fixed + roaffine

            else:
                self.roaffine = self.fixed.to_affine()

            return self.roaffine

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).reshape((1, ) * indices.ndim)

        return DecRuleSub(self, indices, item)

    def __neg__(self):

        return - self.to_affine()

    def __add__(self, other):

        return self.to_affine().__add__(other)

    def __sub__(self, other):

        return self.to_affine().__sub__(other)

    def __rsub__(self, other):

        return self.to_affine().__rsub__(other)

    def __mul__(self, other):

        check_numeric(other)

        return self.to_affine().__mul__(other)

    def __rmul__(self, other):

        check_numeric(other)

        return self.to_affine().__rmul__(other)

    def __matmul__(self, other):

        check_numeric(other)

        return self.to_affine().__matmul__(other)

    def __rmatmul__(self, other):

        check_numeric(other)

        return self.to_affine().__rmatmul__(other)

    def sum(self, axis=None):

        return self.to_affine().sum(axis)

    def __le__(self, other):

        return (self - other).__le__(0)

    def __ge__(self, other):

        return (other - self).__le__(0)

    def get(self, rvar=None):

        if rvar is None:
            return self.fixed.get()
        else:
            if rvar.model.mtype != 'S':
                ValueError('The input is not a random variable.')
            ldr_coeff = np.zeros((self.size,
                                  self.model.rc_model.vars[-1].last))
            rand_ind = rvar.get_ind()
            row_ind, col_ind = np.where(self.depend == 1)
            ldr_coeff[row_ind, col_ind] = self.var_coeff.get()

            size = rvar.to_affine().size
            return ldr_coeff[:, rand_ind].reshape((size, ) + self.shape)


class DecRuleSub:

    __array_priority__ = 105

    def __init__(self, dec_rule, indices, item):

        self.dec_rule = dec_rule
        self.shape = indices.shape
        self.indices = indices.flatten()
        self.item = item

    def adapt(self, rvar):

        self.dec_rule.adapt(rvar, self.indices)

    def to_affine(self):

        roaffine = self.dec_rule.to_affine()
        raffine = roaffine.raffine[self.indices, :]
        affine = roaffine.affine[self.item]

        return RoAffine(raffine, affine, self.dec_rule.model.sup_model)

    def __neg__(self):

        return - self.to_affine()

    def __add__(self, other):

        return self.to_affine().__add__(other)

    def __sub__(self, other):

        return self.to_affine().__sub__(other)

    def __rsub__(self, other):

        return self.to_affine().__rsub__(other)

    def __mul__(self, other):

        check_numeric(other)

        return self.to_affine().__mul__(other)

    def __rmul__(self, other):

        check_numeric(other)

        return self.to_affine().__rmul__(other)

    def __matmul__(self, other):

        check_numeric(other)

        return self.to_affine().__matmul__(other)

    def __rmatmul__(self, other):

        check_numeric(other)

        return self.to_affine().__rmatmul__(other)

    def sum(self, axis=None):

        return self.to_affine().sum(axis)

    def __le__(self, other):

        return (self - other).__le__(0)

    def __ge__(self, other):

        return (other - self).__le__(0)

from .subroutines import *
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
from numbers import Real
from scipy.sparse import csr_matrix
from collections import Iterable


class Model:
    """
    The Model class creates an LP model object
    """

    def __init__(self, nobj=False, mtype='R', name=None):

        self.mtype = mtype
        self.nobj = nobj
        self.name = name

        self.vars = []
        self.auxs = []
        self.last = 0
        self.lin_constr = []
        self.pws_constr = []
        self.bounds = []
        self.aux_constr = []
        self.aux_bounds = []
        self.obj = None
        self.sign = 1
        self.primal = None
        self.dual = None
        self.solution = None
        self.pupdate = True
        self.dupdate = True

        if not nobj:
            self.dvar()

    def dvar(self, shape=(1,), vtype='C', name=None, aux=False):

        if not isinstance(shape, tuple):
            shape = (int(shape), )
        new_var = Vars(self, self.last, shape, vtype, name)
        if not aux:
            self.vars.append(new_var)
        else:
            self.auxs.append(new_var)
        self.last += np.prod(shape)
        return new_var

    def st(self, constr):

        if isinstance(constr, Iterable):
            for item in constr:
                self.st(item)
        else:
            if constr.model is not self:
                raise ValueError('Constraints are not defined for this model.')
            if isinstance(constr, LinConstr):
                self.lin_constr.append(constr)
            elif isinstance(constr, CvxConstr):
                if constr.xtype in 'AMI':
                    self.pws_constr.append(constr)
                else:
                    raise TypeError('Incorrect constraint type.')
            elif isinstance(constr, Bounds):
                self.bounds.append(constr)
            else:
                raise TypeError('Unknown constraint type.')

        self.pupdate = True
        self.dupdate = True

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

    def do_math(self, primal=True, refresh=True, obj=False):

        if primal:
            if self.primal is not None and not self.pupdate:
                return self.primal

            if refresh:
                self.auxs = []
                self.aux_constr = []
                self.aux_bounds = []
                self.last = self.vars[-1].first + self.vars[-1].size

            more_cvx = []
            if self.obj:
                obj_constr = (self.vars[0] >= self.sign * self.obj)
                if isinstance(obj_constr, LinConstr):
                    self.aux_constr.append(obj_constr)
                elif isinstance(obj_constr, CvxConstr):
                    more_cvx.append(obj_constr)

            for constr in self.pws_constr + more_cvx:
                if constr.xtype == 'A':
                    self.aux_constr.append(constr.affine_in +
                                           constr.affine_out <= 0)
                    self.aux_constr.append(-constr.affine_in +
                                           constr.affine_out <= 0)
                elif constr.xtype == 'M':
                    aux = self.dvar(constr.affine_in.shape, aux=True)
                    self.aux_constr.append(constr.affine_in <= aux)
                    self.aux_constr.append(-constr.affine_in <= aux)
                    self.aux_constr.append(sum(aux) + constr.affine_out <= 0)
                elif constr.xtype == 'I':
                    aux = self.dvar(1, aux=True)
                    self.aux_constr.append(constr.affine_in <= aux)
                    self.aux_constr.append(-constr.affine_in <= aux)
                    self.aux_constr.append(aux + constr.affine_out <= 0)
            if obj:
                obj = np.array(csr_matrix(([1.0], ([0], [0])),
                                          (1, self.last)).todense())
            else:
                obj = np.ones((1, self.last))

            data_list = []
            indices_list = []
            indptr = [0]
            last = 0

            data_list += [item.linear.data
                          for item in self.lin_constr + self.aux_constr]
            indices_list += [item.linear.indices
                             for item in self.lin_constr + self.aux_constr]

            if data_list:
                data = np.concatenate(tuple(data_list))
                indices = np.concatenate(tuple(indices_list))
                for item in self.lin_constr + self.aux_constr:
                    indptr.extend(list(item.linear.indptr[1:] + last))
                    last += item.linear.indptr[-1]

                    linear = csr_matrix((data, indices, indptr),
                                        (len(indptr) - 1, self.last))

                    const_list = [item.const for item in
                                  self.lin_constr + self.aux_constr]

                    sense_list = [item.sense
                                  for item in self.lin_constr
                                  + self.aux_constr]

                const = np.concatenate(tuple(const_list))
                sense = np.concatenate(tuple(sense_list))
            else:
                linear = csr_matrix(([], ([], [])), (1, self.last))
                const = np.array([0])
                sense = np.array([1])

            vtype = np.concatenate([np.array([item.vtype] * item.size)
                                    for item in self.vars + self.auxs])

            # ub = np.array([np.infty] * linear.shape[1])
            # lb = np.array([-np.infty] * linear.shape[1])
            ub = np.array([np.infty] * self.last)
            lb = np.array([-np.infty] * self.last)

            for b in self.bounds + self.aux_bounds:
                if b.btype == 'U':
                    ub[b.indices] = np.minimum(b.values, ub[b.indices])
                elif b.btype == 'L':
                    lb[b.indices] = np.maximum(b.values, lb[b.indices])

            formula = LinProg(linear, const, sense,
                              vtype, ub, lb, obj)
            self.primal = formula
            self.pupdate = False

            return formula

        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

            primal = self.do_math()
            if 'B' in primal.vtype or 'I' in primal.vtype:
                string = '\nIntegers detected.'
                string += '\nDual of the continuous relaxtion is returned'
                warnings.warn(string)

            primal_linear = primal.linear
            primal_const = primal.const
            primal_sense = primal.sense
            indices_ub = np.where((primal.ub != 0) &
                                  (primal.ub != np.infty))[0]
            indices_lb = np.where((primal.lb != 0) &
                                  (primal.lb != - np.infty))[0]

            nub = len(indices_ub)
            nlb = len(indices_lb)
            nv = primal_linear.shape[1]
            if nub > 0:
                matrix_ub = csr_matrix((np.array([1] * nub), indices_ub,
                                        np.arange(nub + 1)), (nub, nv))
                primal_linear = sp.vstack((primal_linear, matrix_ub))
                primal_const = np.concatenate((primal_const,
                                               primal.ub[indices_ub]))
                primal_sense = np.concatenate((primal_sense, np.zeros(nub)))
            if nlb > 0:
                matrix_lb = csr_matrix((np.array([-1] * nlb), indices_lb,
                                        np.arange(nlb + 1)), (nlb, nv))
                primal_linear = sp.vstack((primal_linear, matrix_lb))
                primal_const = np.concatenate((primal_const,
                                               -primal.lb[indices_lb]))
                primal_sense = np.concatenate((primal_sense, np.zeros(nlb)))

            indices_free = np.where((primal.lb != 0) &
                                    (primal.ub != 0))[0]
            indices_neg = np.where(primal.ub == 0)[0]

            dual_linear = csr_matrix(primal_linear.T)
            ndv = dual_linear.shape[1]
            dual_obj = - primal_const
            dual_const = primal.obj.reshape((nv, ))
            dual_sense = np.zeros(dual_linear.shape[0])
            dual_sense[indices_free] = 1
            dual_ub = np.zeros(dual_linear.shape[1])
            dual_lb = - np.ones(ndv) * np.infty

            indices_eq = np.where(primal_sense == 1)[0]
            if len(indices_eq):
                dual_ub[indices_eq] = np.infty

            if len(indices_neg) > 0:
                dual_linear[indices_neg, :] = - dual_linear[indices_neg, :]
                dual_const[indices_neg] = - dual_const[indices_neg]

            formula = LinProg(dual_linear, dual_const, dual_sense,
                              np.array(['C']*ndv), dual_ub, dual_lb, dual_obj)
            self.dual = formula
            self.dupdate = False

            return formula

    def solve(self, solver, display=True, export=False):

        self.solution = solver.solve(self.do_math(obj=True), display, export)

    def get(self):

        if self.solution is None:
            raise SyntaxError('The model is unsolved or no feasible solution.')
        return self.sign * self.solution.objval


class SparseVec:

    __array_priority__ = 200

    def __init__(self, index, value, nvar):

        self.index = index
        self.value = value
        self.nvar = nvar

    def __str__(self):

        string = 'Indices: ' + str(self.index) + ' | '
        string += 'Values: ' + str(self.value)

        return string

    def __repr__(self):

        return self.__str__()

    def __add__(self, other):

        return SparseVec(self.index+other.index,
                         self.value+other.value, max(self.nvar, other.nvar))

    def __radd__(self, other):

        return self.__add__(other)

    def __mul__(self, other):

        values = [v*other for v in self.value]
        return SparseVec(self.index, values, self.nvar)

    def __rmul__(self, other):

        return self.__mul__(other)


class Vars:
    """
    The Var class creates a variable array.
    """

    __array_priority__ = 100

    def __init__(self, model, first, shape, vtype, name, sparray=None):

        self.model = model
        self.first = first
        self.shape = shape
        self.size = int(np.prod(shape))
        self.last = first + self.size
        self.ndim = len(shape)
        self.vtype = vtype
        self.name = name
        self.sparray = sparray

    def __str__(self):

        if self.vtype not in ['C', 'B', 'I']:
            raise ValueError('Unknown variable type.')

        var_name = '' if self.name is None else self.name + ': '
        """
        model_type = ('RSO model' if self.model.mtype == 'R' else
                      'Robust counterpart' if self.model.mtype == 'C' else
                      'Support' if self.model.mtype == 'S' else
                      'Expectation set' if self.model.mtype == 'E' else
                      'Probability set')
        """
        var_type = ('continuous' if self.vtype == 'C' else
                    'binary' if self.vtype == 'B' else 'integer')
        suffix = 's' if np.prod(self.shape) > 1 else ''

        string = var_name
        string += 'x'.join([str(size) for size in self.shape]) + ' '
        string += var_type + ' variable' + suffix
        # string += ' ({0})'.format(model_type)

        return string

    def __repr__(self):

        return self.__str__()

    def sv_array(self, index=False):

        shape = self.shape
        shape = shape if isinstance(shape, tuple) else (int(shape), )
        size = np.prod(shape).item()

        if index:
            elements = [SparseVec([i], [1], size) for i in range(size)]
        else:
            elements = [SparseVec([i], [1.0], size) for i in range(size)]

        return np.array(elements).reshape(shape)

    # noinspection PyPep8Naming
    @property
    def T(self):

        return self.to_affine().T

    def to_affine(self):

        dim = self.size

        data = np.ones(dim)
        indices = self.first + np.arange(dim)
        indptr = np.arange(dim+1)

        linear = csr_matrix((data, indices, indptr),
                            shape=(dim, self.model.last))
        const = np.zeros(self.shape)

        # if self.sparray is None:
        #     self.sparray = sparse_array(self.shape)
        #     self.sparray = self.sv_array()

        return Affine(self.model, linear, const, self.sparray)

    def get_ind(self):

        return np.array(range(self.first, self.first + self.size))

    def reshape(self, shape):

        return self.to_affine().reshape(shape)

    def norm(self, degree):

        return self.to_affine().norm(degree)

    def get(self):

        if self.model.solution is None:
            raise SyntaxError('The model is unsolved.')

        indices = range(self.first, self.first + self.size)
        var_sol = np.array(self.model.solution.x)[indices]
        if isinstance(var_sol, np.ndarray):
            var_sol = var_sol.reshape(self.shape)

        return var_sol

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).reshape((1, ) * self.ndim)

        # if self.sparray is None:
        #     self.sparray = sparse_array(self.shape)
        #     self.sparray = self.sv_array()

        return VarSub(self, indices)

    def __iter__(self):

        shape = self.shape
        for i in range(shape[0]):
            yield self[i]

    def __abs__(self):

        return self.to_affine().__abs__()

    def sum(self, axis=None):

        return self.to_affine().sum(axis)

    def __mul__(self, other):

        return self.to_affine() * other

    def __rmul__(self, other):

        return other * self.to_affine()

    def __matmul__(self, other):

        return self.to_affine() @ other

    def __rmatmul__(self, other):

        return other @ self.to_affine()

    def __add__(self, other):

        return self.to_affine() + other

    def __radd__(self, other):

        return self.to_affine() + other

    def __sub__(self, other):

        return self.to_affine() - other

    def __rsub__(self, other):

        return (-self.to_affine()) + other

    def __neg__(self):

        return - self.to_affine()

    def __le__(self, other):

        if isinstance(other, (Real, np.ndarray)) or sp.issparse(other):
            upper = other + np.zeros(self.shape)
            upper = upper.reshape((upper.size, ))
            indices = np.arange(self.first, self.first + self.size,
                                dtype=np.int32)
            return Bounds(self.model, indices, upper, 'U')
        else:
            return self.to_affine() <= other

    def __ge__(self, other):

        if isinstance(other, (Real, np.ndarray)) or sp.issparse(other):
            lower = other + np.zeros(self.shape)
            lower = lower.reshape((lower.size, ))
            indices = np.arange(self.first, self.first + self.size,
                                dtype=np.int32)
            return Bounds(self.model, indices, lower, 'L')
        else:
            return self.to_affine() >= other

    def __eq__(self, other):

        return self.to_affine() == other


class VarSub(Vars):
    """
    The VarSub class creates a variable array with subscript indices
    """

    def __init__(self, var, indices):

        super().__init__(var.model, var.first,
                         var.shape, var.vtype, var.name, var.sparray)
        self.indices = indices

    def __repr__(self):

        var_name = '' if self.name is None else self.name + ': '
        """
        model_type = ('RSO model' if self.model.mtype == 'R' else
                      'Robust counterpart' if self.model.mtype == 'C' else
                      'Support' if self.model.mtype == 'S' else
                      'Expectation set' if self.model.mtype == 'E' else
                      'Probability set')
        """
        var_type = ('continuous' if self.vtype == 'C' else
                    'binary' if self.vtype == 'B' else 'integer')
        suffix = 's' if np.prod(self.shape) > 1 else ''

        string = var_name
        string += 'x'.join([str(dim) for dim in self.indices.shape]) + ' '
        string += 'slice of '
        string += var_type + ' variable' + suffix
        # string += ' ({0})'.format(model_type)

        return string

    @property
    def T(self):

        return self.to_affine().T

    def get_ind(self):

        indices_all = super().get_ind()
        return indices_all[self.indices]

    def __getitem__(self, item):

        raise SyntaxError('Nested indexing of variables is forbidden.')

    def sum(self, axis=None):

        return self.to_affine().sum(axis)

    def to_affine(self):

        select = list(self.indices.reshape((self.indices.size,)))

        dim = self.size
        data = np.ones(dim)
        indices = self.first + np.arange(dim)
        indptr = np.arange(dim + 1)

        linear = csr_matrix((data, indices, indptr),
                            shape=(dim, self.model.last))
        const = np.zeros(self.indices.shape)

        return Affine(self.model, linear[select, :], const)

    def reshape(self, shape):

        return self.to_affine().reshape(shape)

    def __add__(self, other):

        return self.to_affine() + other

    def __radd__(self, other):

        return self.to_affine() + other

    def __le__(self, other):

        if not isinstance(other, (Real, np.ndarray)):
            return self.to_affine().__le__(other)

        upper = super().__le__(other)
        indices = self.indices.reshape((self.indices.size, ))
        bound_indices = upper.indices.reshape((upper.indices.size, ))[indices]
        bound_values = upper.values.reshape(upper.values.size)[indices]

        return Bounds(upper.model, bound_indices, bound_values, 'U')

    def __ge__(self, other):

        if not isinstance(other, (Real, np.ndarray)):
            return self.to_affine().__ge__(other)

        lower = super().__ge__(other)
        indices = self.indices.reshape((self.indices.size, ))
        bound_indices = lower.indices.reshape((lower.indices.size, ))[indices]
        bound_values = lower.values.reshape((lower.indices.size, ))[indices]

        return Bounds(lower.model, bound_indices, bound_values, 'L')


class Affine:
    """
    The Affine class creates an array of affine expressions
    """

    __array_priority__ = 100

    def __init__(self, model, linear, const, sparray=None):

        self.model = model
        self.linear = linear
        self.const = const
        self.shape = const.shape
        self.size = np.prod(self.shape)
        self.sparray = sparray
        self.expect = False

    def __repr__(self):

        """
        model_type = ('RSO model' if self.model.mtype == 'R' else
                      'Robust counterpart' if self.model.mtype == 'C' else
                      'Support' if self.model.mtype == 'S' else
                      'Expectation set' if self.model.mtype == 'E' else
                      'Probability set')
        """
        string = 'x'.join([str(dim) for dim in self.shape]) + ' '
        string += 'affine expressions '
        # string += '({0})'.format(model_type)

        return string

    def __getitem__(self, item):

        if self.sparray is None:
            # self.sparray = sparse_array(self.shape)
            self.sparray = self.sv_array()

        indices = self.sparray[item]
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).reshape((1, ))

        # linear = array_to_sparse(indices) @ self.linear
        linear = sv_to_csr(indices) @ self.linear
        const = self.const[item]
        if not isinstance(const, np.ndarray):
            const = np.array([const])

        return Affine(self.model, linear, const)

    def to_affine(self):

        return self

    def rand_to_roaffine(self, rc_model):

        size = self.size
        num_rand = self.model.vars[-1].last
        reduced_linear = self.linear[:, :num_rand]
        num_dec = rc_model.last

        raffine = Affine(rc_model,
                         csr_matrix((size*num_rand, num_dec)),
                         reduced_linear.toarray())
        affine = Affine(rc_model, csr_matrix((size, num_dec)),
                        self.const)

        return RoAffine(raffine, affine, self.model)

    def sv_array(self, index=False):

        shape = self.shape
        shape = shape if isinstance(shape, tuple) else (int(shape), )
        size = np.prod(shape).item()

        if index:
            elements = [SparseVec([i], [1], size) for i in range(size)]
        else:
            elements = [SparseVec([i], [1.0], size) for i in range(size)]

        return np.array(elements).reshape(shape)

    def sv_zeros(self, nvar):

        shape = (self.shape if isinstance(self.shape, tuple) else
                 (int(self.shape),))
        size = np.prod(self.shape).item()
        elements = [SparseVec([], [], nvar) for _ in range(size)]

        return np.array(elements).reshape(shape)

    # noinspection PyPep8Naming
    @property
    def T(self):


        """
        if self.sparray is None:
            # self.sparray = sparse_array(self.shape)
            self.sparray = self.sv_array()

        trans_sparray = self.sparray.T
        # linear = array_to_sparse(trans_sparray) @ self.linear
        linear = sv_to_csr(trans_sparray) @ self.linear
        """
        linear = sp_trans(self) @ self.linear
        const = self.const.T

        return Affine(self.model, linear, const)

    def reshape(self, shape):

        new_const = self.const.reshape(shape)
        return Affine(self.model, self.linear, new_const)

    def sum(self, axis=None):

        if self.sparray is None:
            # self.sparray = sparse_array(self.shape)
            self.sparray = self.sv_array()

        indices = self.sparray.sum(axis=axis)
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices])

        # linear = array_to_sparse(indices) @ self.linear
        linear = sv_to_csr(indices) @ self.linear
        const = self.const.sum(axis=axis)
        if not isinstance(const, np.ndarray):
            const = np.array([const])

        return Affine(self.model, linear, const)

    def __abs__(self):

        return Convex(self, np.zeros(self.shape), 'A', 1)

    def abs(self):

        return self.__abs__()

    def norm(self, degree):

        shape = self.shape
        if np.prod(shape) != max(shape):
            raise ValueError('Funciton "norm" only applies to vectors.')

        new_shape = (1,) * len(shape)
        if degree == 1:
            return Convex(self, np.zeros(new_shape), 'M', 1)
        elif degree == np.infty or degree == 'inf':
            return Convex(self, np.zeros(new_shape), 'I', 1)
        elif degree == 2:
            return Convex(self, np.zeros(new_shape), 'E', 1)
        else:
            raise ValueError('Incorrect degree for the norm function.')

    def square(self):

        size = self.size
        shape = self.shape

        return Convex(self.reshape((size,)), np.zeros(shape), 'S', 1)

    def sumsqr(self):

        shape = self.shape
        if np.prod(shape) != max(shape):
            raise ValueError('Funciton "sumsqr" only applies to vectors.')

        new_shape = (1,) * len(shape)
        return Convex(self, np.zeros(new_shape), 'Q', 1)

    def __mul__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            if self.model.mtype == 'R' and other.model.mtype == 'S':
                """
                affine = self
                raffine = affine.reshape((affine.size, 1))
                other = other.to_affine()
                # raffine = raffine * np.array(other.linear.todense())

                rvar_last = other.model.vars[-1].last
                reduced_linear = other.linear[:, :rvar_last]

                trans_sparray = (np.ones(affine.size) *  ####################
                                 np.array([line for line in reduced_linear]))

                raffine = raffine * array_to_sparse(trans_sparray)
                """
                other = other.to_affine()
                if self.shape != other.shape:
                    raffine = self * np.ones(other.to_affine().shape)
                    other = np.ones(self.shape) * other.to_affine()
                else:
                    raffine = self
                    other = other.to_affine()

                raffine = raffine.reshape((raffine.size, 1))

                rvar_last = other.model.vars[-1].last
                reduced_linear = other.linear[:, :rvar_last]
                trans_sparray = np.array([line for line in reduced_linear])

                raffine = raffine * array_to_sparse(trans_sparray)
                affine = self * other.const

                return RoAffine(raffine, affine, other.model)
            else:
                return other.__mul__(self)

        else:
            other = check_numeric(other)

            if isinstance(other, Real):
                other = np.array([other])

            """
            if self.sparray is None:
                self.sparray = sparse_array(self.shape)

            new_const = self.const * other

            new_sparray = self.sparray * other
            new_linear = array_to_sparse(new_sparray) @ self.linear
            """

            """
            new_const = self.const * other

            svarray = self.sv_array()
            new_svarray = svarray * other
            new_linear = sv_to_csr(new_svarray) @ self.linear
            """
            new_linear = sparse_mul(other, self.to_affine()) @ self.linear
            new_const = self.const * other

            return Affine(self.model, new_linear, new_const)

    def __rmul__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            if self.model.mtype == 'R' and other.model.mtype == 'S':
                """
                affine = self
                raffine = affine.reshape((affine.size, 1))
                other = other.to_affine()
                # raffine = np.array(other.linear.todense()) * raffine

                rvar_last = other.model.vars[-1].last
                reduced_linear = other.linear[:, :rvar_last]
                trans_sparray = (np.array([line for line in reduced_linear]) *
                                 np.ones(affine.size))
                """
                other = other.to_affine()
                if self.shape != other.shape:
                    raffine = self * np.ones(other.to_affine().shape)
                    other = np.ones(self.shape) * other.to_affine()
                else:
                    raffine = self
                    other = other.to_affine()

                raffine = raffine.reshape((raffine.size, 1))

                rvar_last = other.model.vars[-1].last
                reduced_linear = other.linear[:, :rvar_last]
                trans_sparray = np.array([line for line in reduced_linear])

                raffine = raffine * array_to_sparse(trans_sparray)
                affine = self * other.const

                return RoAffine(raffine, affine, other.model)
            else:
                return other.__rmul__(self)
        else:
            other = check_numeric(other)

            if isinstance(other, Real):
                other = np.array([other])

            """
            new_const = other * self.const

            svarray = self.sv_array()
            new_svarray = other * svarray
            new_linear = sv_to_csr(new_svarray) @ self.linear
            """

            new_linear = sparse_mul(other, self.to_affine()) @ self.linear
            new_const = self.const * other

            return Affine(self.model, new_linear, new_const)

    def __matmul__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            if self.model.mtype == 'R' and other.model.mtype == 'S':

                other = other.to_affine()
                affine = self @ other.const

                ind_array = self.sv_array()
                temp = ind_array @ np.arange(other.size).reshape(other.shape)
                if isinstance(temp, np.ndarray):
                    all_items = list(temp.reshape((temp.size, )))
                else:
                    all_items = [temp]
                    temp = np.array([temp])
                col_ind = np.concatenate(tuple(item.index
                                               for item in all_items))
                row_ind = tuple(np.array(all_items[i].value) + i*other.size
                                for i in range(len(all_items)))
                row_ind = np.concatenate(row_ind)
                csr_temp = csr_matrix((np.ones(len(col_ind)),
                                       (row_ind, col_ind)),
                                      shape=(temp.size*other.size, self.size))
                self_flat = self.reshape(self.size)
                affine_temp = (csr_temp @ self_flat).reshape((temp.size,
                                                              other.size))
                raffine = affine_temp @ other.linear

                return RoAffine(raffine, affine, other.model)
            elif self.model.mtype == 'S' and other.model.mtype == 'R':

                affine = self.const @ other
                other = other.to_affine()

                """
                temp = np.ones((self.size, affine.size))
                temp = (other * temp).reshape(temp.shape)
                raffine = temp @ self.linear
                """

                ind_array = self.sv_array()
                temp = ind_array @ np.arange(other.size).reshape(other.shape)
                if isinstance(temp, np.ndarray):
                    all_items = list(temp.reshape((temp.size, )))
                else:
                    all_items = [temp]
                    temp = np.array([temp])
                col_ind = np.concatenate(tuple(item.value
                                               for item in all_items))
                row_ind = tuple(np.array(all_items[i].index) + i*self.size
                                for i in range(len(all_items)))
                row_ind = np.concatenate(row_ind)
                csr_temp = csr_matrix((np.ones(len(col_ind)),
                                       (row_ind, col_ind)),
                                      shape=(temp.size*self.size, other.size))
                other_flat = other.reshape(other.size)
                affine_temp = (csr_temp @ other_flat).reshape((temp.size,
                                                               self.size))
                raffine = affine_temp @ self.linear

                return RoAffine(raffine, affine, self.model)
        else:
            other = check_numeric(other)

            new_const = self.const @ other
            if not isinstance(new_const, np.ndarray):
                new_const = np.array([new_const])

            """
            svarray = self.sv_array()
            new_svarray = svarray @ other
            if not isinstance(new_svarray, np.ndarray):
                new_svarray = np.array([new_svarray])
            new_linear = sv_to_csr(new_svarray) @ self.linear
            """

            new_linear = sp_lmatmul(other, self, new_const.shape) @ self.linear

            return Affine(self.model, new_linear, new_const)

    def __rmatmul__(self, other):

        other = check_numeric(other)

        new_const = other @ self.const
        if not isinstance(new_const, np.ndarray):
            new_const = np.array([new_const])
        """
        svarray = self.sv_array()
        new_svarray = other @ svarray
        if not isinstance(new_svarray, np.ndarray):
            new_svarray = np.array([new_svarray])
        new_linear = sv_to_csr(new_svarray) @ self.linear
        """

        new_linear = sp_matmul(other, self, new_const.shape) @ self.linear

        return Affine(self.model, new_linear, new_const)

    def __add__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            if isinstance(other, (Vars, VarSub)):
                other = other.to_affine()

            if self.model.mtype != other.model.mtype:
                if self.model.mtype == 'R':
                    return other.rand_to_roaffine(self.model).__add__(self)
                elif other.model.mtype == 'R':
                    return self.rand_to_roaffine(other.model).__add__(other)
                else:
                    raise ValueError('Models mismatch.')

            new_const = other.const + self.const

            if self.shape == other.shape:
                new_linear = add_linear(self.linear, other.linear)
            else:
                """
                left_sparray = self.sv_array()
                right_sparray = other.sv_array()

                left_zero = self.sv_zeros(other.size)
                right_zero = other.sv_zeros(self.size)

                left_sparse = sv_to_csr(left_sparray + right_zero)
                right_sparse = sv_to_csr(left_zero + right_sparray)

                left_linear = left_sparse @ self.linear
                right_linear = right_sparse @ other.linear
                """

                left_linear = (self * np.ones(other.shape)).linear
                right_linear = (other * np.ones(self.shape)).linear

                new_linear = add_linear(left_linear, right_linear)
        elif isinstance(other, np.ndarray):
            other = check_numeric(other)
            new_const = other + self.const

            if self.shape == other.shape:
                new_linear = self.linear
            else:
                if self.shape != other.shape:
                    # sparray = self.sv_array()
                    # zero = self.sv_zeros(other.size)
                    # sparse = sv_to_csr(sparray + zero)

                    new_linear = (self*np.ones(other.shape)).linear
                else:
                    new_linear = self.linear
        elif isinstance(other, Real):
            other = check_numeric(other)
            new_const = other + self.const
            new_linear = self.linear
        else:
            # raise TypeError('Incorrect data type.')
            return other.__add__(self)

        return Affine(self.model, new_linear, new_const)

    def __radd__(self, other):

        return self + other

    def __neg__(self):

        return Affine(self.model, -self.linear, -self.const)

    def __sub__(self, other):

        return self + (-other)

    def __rsub__(self, other):

        return (-self) + other

    def __le__(self, other):

        left = self - other
        if isinstance(left, Affine):
            return LinConstr(left.model, left.linear,
                             -left.const.reshape((left.const.size, )),
                             np.zeros(left.const.size))
        else:
            return left.__le__(0)

    def __ge__(self, other):

        left = other - self
        if isinstance(left, Affine):
            return LinConstr(left.model, left.linear,
                             -left.const.reshape((left.const.size,)),
                             np.zeros(left.const.size))
        else:
            return left.__le__(0)

    def __eq__(self, other):

        left = self - other
        return LinConstr(left.model, left.linear,
                         -left.const.reshape((left.const.size,)),
                         np.ones(left.const.size))


class Convex:
    """
    The Convex class creates an object of convex functions
    """

    def __init__(self, affine_in, affine_out, xtype, sign):

        self.model = affine_in.model
        self.affine_in = affine_in
        self.affine_out = affine_out
        self.size = affine_out.size
        self.xtype = xtype
        self.sign = sign

    def __repr__(self):
        xtypes = {'A': 'Absolute functions',
                  'M': 'One-norm functions',
                  'E': 'Eclidean norm functions',
                  'I': 'Infinity norm functions',
                  'S': 'Element-wise square functions',
                  'Q': 'Quadratic functions'}
        shapes = 'x'.join([str(dim) for dim in self.affine_out.shape])
        string = shapes + ' ' + xtypes[self.xtype]

        return string

    def __str__(self):

        return self.__repr__()

    def __neg__(self):

        return Convex(self.affine_in, -self.affine_out, self.xtype, -self.sign)

    def __add__(self, other):

        affine_in = self.affine_in
        affine_out = self.affine_out + other
        if not isinstance(affine_out,
                          (Vars, VarSub, Affine, Real, np.ndarray)):
            raise TypeError('Incorrect data types.')

        new_convex = Convex(affine_in, affine_out, self.xtype, self.sign)

        return new_convex

    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):

        return self.__add__(-other)

    def __rsub__(self, other):

        return (-self).__add__(other)

    def __mul__(self, other):

        if not isinstance(other, Real):
            raise SyntaxError('Incorrect syntax.')

        if self.xtype in 'AMIE':
            multiplier = abs(other)
        elif self.xtype in 'SQ':
            multiplier = abs(other) ** 0.5
        else:
            raise ValueError('Unknown type of convex function.')
        return Convex(multiplier * self.affine_in, other * self.affine_out,
                      self.xtype, np.sign(other)*self.sign)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __le__(self, other):

        left = self - other
        if left.sign == -1:
            raise ValueError('Non-convex constraints.')

        return CvxConstr(left.model, left.affine_in, left.affine_out,
                         left.xtype)

    def __ge__(self, other):

        right = other - self
        if right.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return CvxConstr(right.model, right.affine_in, right.affine_out,
                         right.xtype)


class RoAffine:
    """
    The Roaffine class creats an object of uncertain affine functions
    """

    __array_priority__ = 101

    def __init__(self, raffine, affine, rand_model):

        self.dec_model = raffine.model
        self.rand_model = rand_model
        self.raffine = raffine
        self.affine = affine
        self.shape = affine.shape
        self.ndim = len(affine.shape)
        self.size = affine.size

    def sv_array(self, index=False):

        shape = self.shape
        shape = shape if isinstance(shape, tuple) else (int(shape), )
        size = np.prod(shape).item()

        if index:
            elements = [SparseVec([i], [1], size) for i in range(size)]
        else:
            elements = [SparseVec([i], [1.0], size) for i in range(size)]

        return np.array(elements).reshape(shape)

    def sv_zeros(self, nvar):

        shape = (self.shape if isinstance(self.shape, tuple) else
                 (int(self.shape),))
        size = np.prod(self.shape).item()
        elements = [SparseVec([], [], nvar) for _ in range(size)]

        return np.array(elements).reshape(shape)

    def reshape(self, shape):

        return RoAffine(self.raffine, self.affine.reshape(shape),
                        self.rand_model)

    @property
    def T(self):

        """
        # sparray = sparse_array(self.shape)
        sparray = self.sv_array()
        trans_sparray = sparray.T
        # raffine = array_to_sparse(trans_sparray) @ self.raffine
        raffine = sv_to_csr(trans_sparray) @ self.raffine
        """

        raffine = sp_trans(self) @ self.raffine
        affine = self.affine.T

        return RoAffine(raffine, affine, self.rand_model)

    def __neg__(self):

        return RoAffine(-self.raffine, -self.affine, self.rand_model)

    def __add__(self, other):

        if isinstance(other, RoAffine):
            if other.shape != self.shape:
                left = self + np.zeros(other.shape)
                right = other + np.zeros(self.shape)
            else:
                left = self
                right = other
            raffine = left.raffine + right.raffine
            affine = left.affine + right.affine
            if self.dec_model is not other.dec_model or \
               self.rand_model is not other.rand_model:
                raise ValueError('Models mismatch.')
            return RoAffine(raffine, affine, self.rand_model)
        elif isinstance(other, (Affine, Vars, VarSub)):
            other = other.to_affine()
            if other.model == self.rand_model:
                if other.shape != self.shape:
                    left = self + np.zeros(other.shape)
                    right = other + np.zeros(self.shape)
                else:
                    left = self
                    right = other.to_affine()

                right_term = right.rand_to_roaffine(left.dec_model)
                return left.__add__(right_term)
            elif other.model == self.dec_model:
                # raffine = self.raffine + np.zeros((other.size,
                #                                    self.raffine.shape[1]))
                # affine = self.affine + other

                # sparray = self.sv_array()
                # zero = self.sv_zeros(other.size)
                # sparse = sv_to_csr(sparray + zero)
                # raffine = sparse @ self.raffine

                if other.shape != self.shape:
                    left = self * np.ones(other.shape)
                    right = other + np.zeros(self.shape)
                else:
                    left = self
                    right = other.to_affine()

                raffine = left.raffine
                affine = left.affine + right

                return RoAffine(raffine, affine, self.rand_model)
            else:
                raise TypeError('Unknown model types.')
        elif isinstance(other, (Real, np.ndarray)):
            if isinstance(other, Real):
                other = np.array([other])
            # raffine = self.raffine + np.zeros((other.size,
            #                                   self.raffine.shape[1]))

            if other.shape == self.shape:
                raffine = self.raffine
            else:
                sparray = (np.arange(self.size).reshape(self.shape)
                           + np.zeros(other.shape))
                index = sparray.flatten()
                size = sparray.size
                sparse = csr_matrix(([1]*size, index, np.arange(size+1)),
                                    shape=[size, self.size])
                raffine = sparse @ self.raffine

            """
            sparray = self.sv_array()
            zero = self.sv_zeros(other.size)
            sparse = sv_to_csr(sparray + zero)
            raffine = sparse @ self.raffine
            """

            affine = self.affine + other
            return RoAffine(raffine, affine, self.rand_model)
        else:
            raise SyntaxError('Syntax error.')

    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):

        return self.__add__(-other)

    def __rsub__(self, other):

        return (-self).__add__(other)

    def __mul__(self, other):

        new_affine = self.affine * other
        if isinstance(other, Real):
            other = np.array([other])

        """
        svarray = self.affine.sv_array()
        new_svarray = svarray * other
        if not isinstance(new_svarray, np.ndarray):
            new_svarray = np.array([new_svarray])

        new_raffine = sv_to_csr(new_svarray) @ self.raffine
        """
        new_raffine = sparse_mul(other, self) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __rmul__(self, other):

        new_affine = other * self.affine
        if isinstance(other, Real):
            other = np.array([other])

        """
        svarray = self.affine.sv_array()
        new_svarray = other * svarray
        if not isinstance(new_svarray, np.ndarray):
            new_svarray = np.array([new_svarray])

        new_raffine = sv_to_csr(new_svarray) @ self.raffine
        """

        new_raffine = sparse_mul(other, self) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __matmul__(self, other):

        other = check_numeric(other)

        new_affine = self.affine @ other

        """
        svarray = self.affine.sv_array()
        new_svarray = svarray @ other
        if not isinstance(new_svarray, np.ndarray):
            new_svarray = np.array([new_svarray])

        new_raffine = sv_to_csr(new_svarray) @ self.raffine
        """

        new_raffine = sp_lmatmul(other, self, new_affine.shape) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __rmatmul__(self, other):

        other = check_numeric(other)

        new_affine = other @ self.affine

        """
        svarray = self.affine.sv_array()
        new_svarray = other @ svarray
        if not isinstance(new_svarray, np.ndarray):
            new_svarray = np.array([new_svarray])

        new_raffine = sv_to_csr(new_svarray) @ self.raffine
        """

        new_raffine = sp_matmul(other, self, new_affine.shape) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def sum(self, axis=None):

        new_affine = self.affine.sum(axis=axis)

        svarray = self.affine.sv_array()
        new_svarray = svarray.sum(axis=axis)
        if not isinstance(new_svarray, np.ndarray):
            new_svarray = np.array([new_svarray])

        new_raffine = sv_to_csr(new_svarray) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __le__(self, other):

        left = self - other
        return RoConstr(left, sense=0)

    def __ge__(self, other):

        right = other - self
        return RoConstr(right, sense=0)


class LinConstr:
    """
    The LinConstr class creates an array of linear constraints.
    """

    def __init__(self, model, linear, const, sense, sign=1):

        self.model = model
        self.linear = linear
        self.const = const
        self.sense = sense
        self.sign = sign


class CvxConstr:
    """
    The CvxConstr class creates an object of convex constraints
    """

    def __init__(self, model, affine_in, affine_out, xtype):

        self.model = model
        self.affine_in = affine_in
        self.affine_out = affine_out
        self.xtype = xtype


class Bounds:
    """
    The Bounds class creates an object for upper or lower bounds
    """

    def __init__(self, model, indices, values, btype):

        self.model = model
        self.indices = indices
        self.values = values
        self.btype = btype


class ConeConstr:

    def __init__(self, model, left_var, left_index, right_var, right_index):

        self.model = model
        self.left_var = left_var
        self.right_var = right_var
        self.left_index = left_index
        self.right_index = right_index


class RoConstr:
    """
    The Roaffine class creats an object of uncertain affine functions
    """

    def __init__(self, roaffine, sense):

        self.dec_model = roaffine.dec_model
        self.rand_model = roaffine.rand_model
        self.raffine = roaffine.raffine
        self.affine = roaffine.affine
        self.shape = roaffine.shape
        self.sense = sense
        self.support = None

    def forall(self, *args):

        constraints = []
        for items in args:
            if isinstance(items, Iterable):
                constraints.extend(list(items))
            else:
                constraints.append(items)

        sup_model = constraints[0].model
        sup_model.reset()
        for item in constraints:
            if item.model is not sup_model:
                raise SyntaxError('Models mismatch.')
            sup_model.st(item)

        self.support = sup_model.do_math(primal=False)

        return self

    def le_to_rc(self, support=None):

        num_constr, num_rand = self.raffine.shape
        support = self.support if not support else support
        size_support = support.linear.shape[1]

        dual_var = self.dec_model.dvar((num_constr, size_support))

        constr1 = (dual_var@support.obj +
                   self.affine.reshape(num_constr) <= 0)

        left = dual_var @ support.linear[:num_rand].T
        left = left + self.raffine * support.const[:num_rand]
        sense2 = np.tile(support.sense[:num_rand], num_constr)
        num_rc_constr = left.const.size
        constr2 = LinConstr(left.model, left.linear,
                            -left.const.reshape(num_rc_constr),
                            sense2)

        index_pos = (support.ub == 0)
        bounds = (dual_var[:, index_pos] <= 0)

        if num_rand == support.linear.shape[0]:
            constr_tuple = constr1, constr2, bounds
        else:
            left = dual_var @ support.linear[num_rand:].T
            sense3 = np.tile(support.sense[num_rand:], num_constr)
            num_rc_constr = left.const.size
            constr3 = LinConstr(left.model, left.linear,
                                left.const.reshape(num_rc_constr),
                                sense3)
            constr_tuple = constr1, constr2, constr3, bounds

        for qconstr in support.qmat:
            cone_constr = ConeConstr(self.dec_model, dual_var, qconstr[1:],
                                     dual_var, qconstr[0])
            constr_tuple = constr_tuple + (cone_constr,)

        return constr_tuple


class LinProg:
    """
    The LinProg class creates an object of linear program
    """

    def __init__(self, linear, const, sense, vtype, ub, lb, obj=None):

        self.obj = obj
        self.linear = linear
        self.const = const
        self.sense = sense
        self.vtype = vtype
        self.ub = ub
        self.lb = lb

    def __repr__(self, header=True):

        linear = self.linear
        nc, nb, ni = (sum(self.vtype == 'C'),
                      sum(self.vtype == 'B'),
                      sum(self.vtype == 'I'))
        nineq, neq = sum(self.sense == 0), sum(self.sense == 1)
        nnz = self.linear.indptr[-1]

        if header:
            string = 'Linear program object:\n'
        else:
            string = ''
        string += '=============================================\n'
        string += 'Number of variables:          {0}\n'.format(linear.shape[1])
        string += 'Continuous/binaries/integers: {0}/{1}/{2}\n'.format(nc,
                                                                       nb, ni)
        string += '---------------------------------------------\n'
        string += 'Number of linear constraints: {0}\n'.format(linear.shape[0])
        string += 'Inequalities/equalities:      {0}/{1}\n'.format(nineq, neq)
        string += 'Number of coefficients:       {0}\n'.format(nnz)

        return string

    def showlc(self):

        var_names = ['x{0}'.format(i)
                     for i in range(1, self.linear.shape[1] + 1)]
        constr_names = ['LC{0}'.format(j)
                        for j in range(1, self.linear.shape[0] + 1)]
        table = pd.DataFrame(self.linear.todense(), columns=var_names,
                             index=constr_names)
        table['sense'] = ['==' if sense else '<=' for sense in self.sense]
        table['constant'] = self.const

        return table

    def solve(self, solver):

        return solver.solve(self)


class Solution:

    def __init__(self, objval, x, stats):

        self.objval = objval
        self.x = x
        self.stats = stats

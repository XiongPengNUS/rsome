from .subroutines import sv_to_csr, sp_trans, sparse_mul, sp_lmatmul, sp_matmul
from .subroutines import array_to_sparse, index_array, check_numeric
from .subroutines import add_linear
from .subroutines import event_dict, comb_set, flat
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
import time
import scipy.optimize as opt
from numbers import Real
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.linalg import sqrtm, eigh
from collections.abc import Iterable, Sized
from typing import List


def def_sol(formula, display=True, params={}):

    try:
        if formula.qmat:
            warnings.warn('the LP solver ignnores SOC constriants.')
    except AttributeError:
        pass

    try:
        if formula.xmat:
            warnings.warn('The LP solver ignores exponential cone constriants.')
    except AttributeError:
        pass

    if any(np.array(formula.vtype) != 'C'):
        warnings.warn('Integrality constraints are ignored in the LP solver. ')

    indices_eq = (formula.sense == 1)
    indices_ineq = (formula.sense == 0)
    linear_eq = formula.linear[indices_eq, :] if len(indices_eq) else None
    linear_ineq = formula.linear[indices_ineq, :] if len(indices_ineq) else None
    const_eq = formula.const[indices_eq] if len(indices_eq) else None
    const_ineq = formula.const[indices_ineq] if len(indices_ineq) else None

    bounds = [(lb, ub) for lb, ub in zip(formula.lb, formula.ub)]

    default = {'maxiter': 1000000000,
               'sparse': True}

    if display:
        print('Being solved by the default LP solver...', flush=True)
        time.sleep(0.2)
    t0 = time.time()
    res = opt.linprog(formula.obj, A_ub=linear_ineq, b_ub=const_ineq,
                      A_eq=linear_eq, b_eq=const_eq,
                      bounds=bounds, options=default)
    stime = time.time() - t0
    if display:
        print('Solution status: {0}'.format(res.status))
        print('Running time: {0:0.4f}s'.format(stime))

    if res.status == 0:
        return Solution(res.x[0], res.x, res.status, stime)
    else:
        status = res.status
        msg = 'The optimal solution can not be found, '
        reasons = ('iteration limit is reached.' if status == 1 else
                   'the problem appears to be infeasible.' if status == 2 else
                   'the problem appears to be unbounded.' if status == 3 else
                   'numerical difficulties encountered.')
        msg += 'because {}'.format(reasons)
        warnings.warn(msg)
        return None


class Model:
    """
    The Model class creates an LP model object
    """

    def __init__(self, nobj=False, mtype='R', name=None, top=None):

        self.mtype = mtype
        self.top = top
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

    def dvar(self, shape=(), vtype='C', name=None, aux=False):

        if not isinstance(shape, tuple):
            shape = (shape, )

        for item in shape:
            if not isinstance(item, (int, np.int8, np.int16,
                                     np.int32, np.int64)):
                raise TypeError('Shape values must be integers!')
        new_shape = tuple(np.array(shape).astype(int))

        vtype = vtype.upper()
        if 'C' not in vtype and 'B' not in vtype and 'I' not in vtype:
            raise ValueError('Unknown variable type.')
        if len(vtype) != 1 and len(vtype) != np.prod(shape):
            raise ValueError('Inconsistent variables and their types.')

        new_var = Vars(self, self.last, new_shape, vtype, name)

        if not aux:
            self.vars.append(new_var)
        else:
            self.auxs.append(new_var)
        self.last += int(np.prod(new_shape))
        return new_var

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
        else:
            if not isinstance(constr, (LinConstr, CvxConstr, Bounds)):
                raise TypeError('Unknown constraint type.')
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

        self.pupdate = True
        self.dupdate = True

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

        if not isinstance(obj, Real):
            if isinstance(obj, VarSub):
                if obj.indices.size > 1:
                    raise ValueError('Incorrect function dimension.')
            else:
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

        if not isinstance(obj, Real):
            if isinstance(obj, VarSub):
                if obj.indices.size > 1:
                    raise ValueError('Incorrect function dimension.')
            else:
                if obj.size > 1:
                    raise ValueError('Incorrect function dimension.')

        self.obj = obj
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def do_math(self, primal=True, refresh=True, obj=True):
        """
        Return the linear programming problem as the standard formula
        of the model.

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
        prog : LinProg
            A linear programming problem.
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
                if isinstance(obj_constr, LinConstr):
                    self.aux_constr.append(obj_constr)
                elif isinstance(obj_constr, CvxConstr):
                    more_cvx.append(obj_constr)

            for constr in self.pws_constr + more_cvx:
                if constr.xtype == 'A':
                    affine_in = constr.affine_in * constr.multiplier
                    self.aux_constr.append(affine_in +
                                           constr.affine_out <= 0)
                    self.aux_constr.append(-affine_in +
                                           constr.affine_out <= 0)
                elif constr.xtype == 'M':
                    affine_in = constr.affine_in * constr.multiplier
                    aux = self.dvar(constr.affine_in.shape, aux=True)
                    self.aux_constr.append(affine_in <= aux)
                    self.aux_constr.append(-affine_in <= aux)
                    self.aux_constr.append(sum(aux) + constr.affine_out <= 0)
                elif constr.xtype == 'I':
                    affine_in = constr.affine_in * constr.multiplier
                    aux = self.dvar(1, aux=True)
                    self.aux_constr.append(affine_in <= aux)
                    self.aux_constr.append(-affine_in <= aux)
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
                                  if isinstance(item.sense, np.ndarray) else
                                  np.array([item.sense])
                                  for item in self.lin_constr
                                  + self.aux_constr]

                const = np.concatenate(tuple(const_list))
                sense = np.concatenate(tuple(sense_list))
            else:
                linear = csr_matrix(([], ([], [])), (1, self.last))
                const = np.array([0])
                sense = np.array([1])

            vtype = np.concatenate([np.array([item.vtype] * item.size)
                                    if len(item.vtype) == 1
                                    else np.array(list(item.vtype))
                                    for item in self.vars + self.auxs])

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

            primal = self.do_math(obj=obj)
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
            # dual_lb = - np.ones(ndv) * np.infty
            dual_lb = - np.array([np.infty] * ndv)

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

    def solve(self, solver=None, display=True, params={}):
        """
        Solve the model with the selected solver interface.

        Parameters
        ----------
            solver : {None, lpg_solver, clp_solver, ort_solver, eco_solver
                      cpx_solver, grb_solver, msk_solver}
                Solver interface used for model solution. Use default solver
                if solver=None.
            display : bool
                Display option of the solver interface.
            params : dict
                A dictionary that specifies parameters of the selected solver.
                So far the argument only applies to Gurobi, CPLEX,and MOSEK.
        """

        if solver is None:
            solution = def_sol(self.do_math(obj=True), display, params)
        else:
            solution = solver.solve(self.do_math(obj=True), display, params)

        if isinstance(solution, Solution):
            self.solution = solution
        else:
            self.solution = None

    def get(self):
        """
        Return the optimal objective value of the solved model.

        Notes
        -----
        An error message is given if the model is unsolved or no solution
        is obtained.
        """

        if self.solution is None:
            raise RuntimeError('The model is unsolved or no solution is obtained.')
        return self.sign * self.solution.objval

    def optimal(self):

        return self.solution is not None


class SparseVec:

    __array_priority__ = 200

    def __init__(self, index, value, nvar):

        self.index = index
        self.value = value
        self.nvar = nvar

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

    def __repr__(self):

        vtype = self.vtype

        var_name = ('' if self.name is None else
                    f'E({self.name}): ' if self.model.mtype == 'E' else
                    f'{self.name}: ')
        var_type = (' continuous' if vtype == 'C' else
                    ' binary' if vtype == 'B' else
                    ' integer' if vtype == 'I' else
                    ' mixed-type')
        suffix = 's' if np.prod(self.shape) > 1 else ''

        mtype = (' decision' if self.model.mtype == 'R' else
                 ' random' if self.model.mtype == 'S' else
                 ' probability' if self.model.mtype == 'P' else
                 ' expectation of random' if self.model.mtype == 'E' else
                 ' unknown')
        var_type = var_type if mtype == ' decision' else ''
        if self.shape == ():
            num = 'an' if (var_type + mtype)[0:2] in [' i', ' a', ' e'] else 'a'
        else:
            num = 'x'.join([str(size) for size in self.shape])

        string = '{}{}{}{} variable{}'.format(var_name, num, var_type, mtype,
                                              suffix)
        return string

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

        return Affine(self.model, linear, const, self.sparray)

    def get_ind(self):

        return np.array(range(self.first, self.first + self.size))

    def reshape(self, shape):

        return self.to_affine().reshape(shape)

    def abs(self):

        return self.to_affine().abs()

    def norm(self, degree):
        """
        Return the first, second, or infinity norm of a 1-D array.

        Refer to `rsome.norm` for full documentation

        See Also
        --------
        rso.norm : equivalent function
        """

        return self.to_affine().norm(degree)

    def square(self):
        """
        Return the element-wise square of an array.

        Refer to `rsome.square` for full documentation

        See Also
        --------
        rso.square : equivalent function
        """

        return self.to_affine().square()

    def sumsqr(self):
        """
        Return the sum of squares of a 1-D array.

        Refer to `rsome.sumsqr` for full documentation.

        See Also
        --------
        rso.sumsqr : equivalent function
        """

        return self.to_affine().sumsqr()

    def quad(self, qmat):
        """
        Return the quadratic expression var @ qmat @ var.

        Refer to `rsome.quad` for full documentation.

        See Also
        --------
        rso.quad : equivalent function
        """

        return self.to_affine().quad(qmat)

    def expcone(self, x, z):
        """
        Return the exponential cone constraint z*exp(x/z) <= var

        Refer to `rsome.expcone` for full documentation.

        See Also
        --------
        rso.expcone : equivalent function
        """

        return self.to_affine().expcone(x, z)

    def exp(self):
        """
        Return the natural exponential function exp(var)

        Refer to `rsome.exp` for full documentation.

        See Also
        --------
        rso.exp : equivalent function
        """

        return self.to_affine().exp()
    
    def pexp(self, scale):
        """
        Return the perspective natural exponential function 
        scale * exp(var/scale)

        Refer to `rsome.pexp` for full documentation.

        See Also
        --------
        rso.pexp : equivalent function
        """

        return self.to_affine().pexp(scale)

    def log(self):
        """
        Return the natural logarithm function log(var)

        Refer to `rsome.log` for full documentation.

        See Also
        --------
        rso.log : equivalent function
        """

        return self.to_affine().log()
    
    def plog(self, scale):
        """
        Return the perspective of natural logarithm function 
        scale * log(var/scale)

        Refer to `rsome.plog` for full documentation.

        See Also
        --------
        rso.plog : equivalent function
        """

        return self.to_affine().plog(scale)

    def entropy(self):
        """
        Return the natural exponential function -sum(var*log(var))

        Refer to `rsome.entropy` for full documentation.

        See Also
        --------
        rso.entropy : equivalent function
        """

        return self.to_affine().entropy()

    def kldiv(self, phat, r):
        """
        Return the KL divergence constraints sum(var*log(var/phat)) <= r

        Refer to `rsome.kldiv` for full documentation.

        See Also
        --------
        rso.kldiv : equivalent function
        """

        return self.to_affine().kldiv(phat, r)

    def get(self):
        """
        Return the optimal solution of the decision variable.

        Notes
        -----
        The optimal solution is returned as a NumPy array with the
        same shape as the defined decision variable.
        An error message is raised if:
        1. The variable is not a decision variable.
        2. The model is unsolved, or no solution is obtained due to
        infeasibility, unboundedness, or numeric issues.
        """

        if self.model.mtype not in 'VR':
            raise TypeError('Not a decision variable.')

        if self.model.solution is None:
            raise RuntimeError('The model is unsolved or no solution is obtained.')

        indices = range(self.first, self.first + self.size)
        var_sol = np.array(self.model.solution.x)[indices]
        if self.shape == ():
            var_sol = var_sol[0]
        else:
            var_sol = var_sol.reshape(self.shape)

        return var_sol

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]

        return VarSub(self, indices)

    def iter(self):

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

        if ((isinstance(other, (Real, np.ndarray)) or sp.issparse(other))
                and self.model.mtype not in 'EP'):
            upper = other + np.zeros(self.shape)
            upper = upper.reshape((upper.size, ))
            indices = np.arange(self.first, self.first + self.size,
                                dtype=np.int32)
            return Bounds(self.model, indices, upper, 'U')
        else:
            return self.to_affine() <= other

    def __ge__(self, other):

        if ((isinstance(other, (Real, np.ndarray)) or sp.issparse(other))
                and self.model.mtype not in 'EP'):
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

        var_name = '' if not self.name else 'slice of {}: '.format(self.name)
        var_type = ('continuous' if self.vtype == 'C' else
                    'binary' if self.vtype == 'B' else 'integer')

        mtype = (' decision' if self.model.mtype == 'R' else
                 'random' if self.model.mtype == 'S' else
                 'probability' if self.model.mtype == 'P' else
                 'expectation of random' if self.model.mtype == 'E' else
                 'unknown')
        var_type = var_type if mtype == ' decision' else ''
        if isinstance(self.indices, np.ndarray):
            num = 'x'.join([str(dim) for dim in self.indices.shape])
            size = np.prod(self.indices.shape)
        else:
            num = 'an' if (var_type + mtype)[0:2] in [' i', ' a', ' e'] else 'a'
            size = 1
        suffix = 's' if size > 1 else ''

        string = '{}{} {}{} variable{}'.format(var_name, num, var_type, mtype,
                                               suffix)
        return string

    @property
    def T(self):

        return self.to_affine().T

    def get_ind(self):

        indices_all = super().get_ind()
        return indices_all[self.indices].flatten()

    def __getitem__(self, item):

        new_indices = self.indices[item]

        return VarSub(self, new_indices)

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

        upper = super().__le__(other)
        if isinstance(upper, Bounds):
            indices = self.indices.reshape((self.indices.size, ))
            bound_indices = upper.indices.reshape((upper.indices.size, ))[indices]
            bound_values = upper.values.reshape(upper.values.size)[indices]

            return Bounds(upper.model, bound_indices, bound_values, 'U')
        else:
            return self.to_affine().__le__(other)

    def __ge__(self, other):

        lower = super().__ge__(other)
        if isinstance(lower, Bounds):
            indices = self.indices.reshape((self.indices.size, ))
            bound_indices = lower.indices.reshape((lower.indices.size, ))[indices]
            bound_values = lower.values.reshape((lower.indices.size, ))[indices]

            return Bounds(lower.model, bound_indices, bound_values, 'L')
        else:
            return self.to_affine().__ge__(other)


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
        self.size = int(np.prod(self.shape))
        self.sparray = sparray
        self.expect = False

    def __repr__(self):

        if self.shape == ():
            string = 'an '
        else:
            string = 'x'.join([str(dim) for dim in self.shape]) + ' '
        suffix = 's' if self.size > 1 else ''
        string += 'affine expression' + suffix
        # string += '({0})'.format(model_type)

        return string

    def __getitem__(self, item):

        if self.sparray is None:
            self.sparray = self.sv_array()

        indices = self.sparray[item]
        linear = sv_to_csr(indices) @ self.linear
        const = self.const[item]

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
        # shape = shape if isinstance(shape, tuple) else (int(shape), )
        size = self.size
        # print(shape)

        if index:
            elements = [SparseVec([i], [1], size) for i in range(size)]
        else:
            elements = [SparseVec([i], [1.0], size) for i in range(size)]

        return np.array(elements).reshape(shape)

    # noinspection PyPep8Naming
    @property
    def T(self):

        linear = sp_trans(self) @ self.linear
        const = self.const.T

        return Affine(self.model, linear, const)

    def reshape(self, shape):

        if isinstance(self.const, np.ndarray):
            new_const = self.const.reshape(shape)
        else:
            new_const = np.array([self.const]).reshape(shape)
        return Affine(self.model, self.linear, new_const)

    def sum(self, axis=None):

        if self.sparray is None:
            # self.sparray = sparse_array(self.shape)
            self.sparray = self.sv_array()

        indices = self.sparray.sum(axis=axis)
        # if not isinstance(indices, np.ndarray):
        #     indices = np.array([indices])

        # linear = array_to_sparse(indices) @ self.linear
        linear = sv_to_csr(indices) @ self.linear
        const = self.const.sum(axis=axis)
        # if not isinstance(const, np.ndarray):
        #     const = np.array([const])

        return Affine(self.model, linear, const)

    def __abs__(self):

        return Convex(self, np.zeros(self.shape), 'A', 1)

    def abs(self):

        return self.__abs__()

    def norm(self, degree):
        """
        Return the first, second, or infinity norm of a 1-D array.

        Refer to `rsome.norm` for full documentation

        See Also
        --------
        rso.norm : equivalent function
        """

        if len(self.shape) != 1:
            err = 'Improper number of dimensions to norm. '
            err += 'The array must be 1-D.'
            raise ValueError(err)

        new_shape = ()
        if degree == 1:
            return Convex(self, np.zeros(new_shape), 'M', 1)
        elif degree == np.infty or degree == 'inf':
            return Convex(self, np.zeros(new_shape), 'I', 1)
        elif degree == 2:
            return Convex(self, np.zeros(new_shape), 'E', 1)
        else:
            raise ValueError('Invalid norm order for the array.')

    def square(self):
        """
        Return the element-wise square of an array.

        Refer to `rsome.square` for full documentation

        See Also
        --------
        rso.square : equivalent function
        """

        size = self.size
        shape = self.shape

        return Convex(self.reshape((size,)), np.zeros(shape), 'S', 1)

    def sumsqr(self):
        """
        Return the sum of squares of a 1-D array.

        Refer to `rsome.sumsqr` for full documentation.

        See Also
        --------
        rso.sumsqr : equivalent function
        """

        shape = self.shape
        if len(shape) != 1:
            err = 'Improper number of dimensions to norm. '
            err += 'The array must be 1-D.'
            raise ValueError(err)

        new_shape = ()
        return Convex(self, np.zeros(new_shape), 'Q', 1)

    def quad(self, qmat):
        """
        Return the quadratic expression affine @ qmat affine.

        Parameters
        ----------
        affine : an array of variables or affine expression
            Input array. The array must be 1-D.
        qmat : a positive or negative semidefinite matrix.

        Returns
        -------
        q : Convex
            The quadratic expression affine @ qmat affine
        """

        if len(self.shape) != 1:
            err = 'Improper number of dimensions to norm. '
            err += 'The array must be 1-D.'
            raise ValueError(err)

        eighvals = eigh(qmat, eigvals_only=True).round(6)
        if all(eighvals >= 0):
            sign = 1
        elif all(eighvals <= 0):
            sign = -1
        else:
            raise ValueError('The input matrix must be semidefinite.')

        sqrt_mat = np.real(sqrtm(sign*qmat))
        affine = sqrt_mat @ self.reshape(self.size)

        if sign == 1:
            return affine.sumsqr()
        else:
            return - affine.sumsqr()

    def expcone(self, x, z):
        """
        Return the exponential cone constraint z*exp(x/z) <= affine

        Refer to `rsome.expcone` for full documentation.

        See Also
        --------
        rso.expcone : equivalent function
        """

        if isinstance(x, (Vars, VarSub)):
            if x.to_affine().size > 1:
                raise ValueError('The expression of x must be a scalar.')
        elif isinstance(x, (Affine, np.ndarray)):
            if x.size > 1:
                raise ValueError('The expression of x must be a scalar')
        if isinstance(x, (Vars, VarSub, Affine)):
            if self.model is not x.model:
                raise ValueError('Models mismatch.')

        if isinstance(z, (Vars, VarSub)):
            if z.to_affine().size > 1:
                raise ValueError('The expression of z must be a scalar.')
        elif isinstance(z, (Affine, np.ndarray)):
            if z.size > 1:
                raise ValueError('The expression of z must be a scalar')
        if isinstance(z, (Vars, VarSub, Affine)):
            if self.model is not z.model:
                raise ValueError('Models mismatch.')

        return ExpConstr(self.model, x, self, z)

    def exp(self):
        """
        Return the natural exponential function exp(affine)

        Refer to `rsome.exp` for full documentation.

        See Also
        --------
        rso.exp : equivalent function
        """

        return Convex(self, np.zeros(self.shape), 'X', 1)
    
    def pexp(self, scale):
        """
        Return the perspective natural exponential function 
        scale * exp(affine/scale)

        Refer to `rsome.pexp` for full documentation.

        See Also
        --------
        rso.pexp : equivalent function
        """

        return PerspConvex(self, scale, np.zeros(self.shape), 'X', 1)

    def log(self):
        """
        Return the natural logarithm function log(affine)

        Refer to `rsome.log` for full documentation.

        See Also
        --------
        rso.log : equivalent function
        """

        return Convex(self, np.zeros(self.shape), 'L', -1)
    
    def plog(self, scale):
        """
        Return the perspective of natural logarithm function 
        scale * log(affine/scale)

        Refer to `rsome.plog` for full documentation.

        See Also
        --------
        rso.plog : equivalent function
        """

        return PerspConvex(self, scale, np.zeros(self.shape), 'L', -1)

    def entropy(self):
        """
        Return the natural exponential function -sum(affine*log(affine))

        Refer to `rsome.entropy` for full documentation.

        See Also
        --------
        rso.entropy : equivalent function
        """

        if self.shape != ():
            if self.size != max(self.shape):
                raise ValueError('The expression must be a vector.')

        return Convex(self, np.float64(0), 'P', -1)

    def kldiv(self, phat, r):
        """
        Return the KL divergence constraints sum(var*log(var/phat)) <= r

        Refer to `rsome.kldiv` for full documentation.

        See Also
        --------
        rso.kldiv : equivalent function
        """

        affine = self.to_affine().reshape((self.size, ))

        if isinstance(phat, Real):
            phat = np.array([phat]*self.size)
        elif isinstance(phat, np.ndarray):
            if phat.size == 1:
                phat = np.array([phat.flatten()[0]] * self.size)
            else:
                phat = phat.reshape(affine.shape)
        elif isinstance(phat, (Vars, VarSub, Affine)):
            if affine.model is not phat.model:
                raise ValueError('Models mismatch.')
            if phat.size == 1:
                phat = phat * np.ones(affine.shape)
            else:
                phat = phat.reshape(affine.shape)

        return KLConstr(affine, phat, r)

    def __mul__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            other = other.to_affine()
            if self.model.mtype == other.model.mtype:
                raise TypeError('Bi-linear expressions are not supported.')
            elif self.model.mtype in 'VR' and other.model.mtype in 'SM':
                if self.model.top is not other.model.top:
                    raise ValueError('Models of operands mismatch.')
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
        elif isinstance(other, (DecRule, DecRuleSub)):
            return self.__mul__(other.to_affine())
        else:
            other = check_numeric(other)
            new_const = self.const * other
            if not isinstance(other, np.ndarray):
                other = np.array([other])
            new_linear = sparse_mul(other, self.to_affine()) @ self.linear

            return Affine(self.model, new_linear, new_const)

    def __rmul__(self, other):

        if isinstance(other, (Real, np.ndarray)) or sp.issparse(other):
            other = check_numeric(other)

            new_const = self.const * other
            if not isinstance(other, np.ndarray):
                other = np.array([other])
            new_linear = sparse_mul(other, self.to_affine()) @ self.linear

            return Affine(self.model, new_linear, new_const)
        else:
            return other.__mul__(self)

    def __matmul__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            other = other.to_affine()
            if self.model.mtype == other.model.mtype:
                raise TypeError('Bi-linear expressions are not supported.')
            elif self.model.mtype in 'VR' and other.model.mtype in 'SM':
                if self.model.top is not other.model.top:
                    raise ValueError('Models of operands mismatch.')
                affine = self @ other.const
                num_rand = other.model.vars[-1].last

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
                raffine = affine_temp @ other.linear[:, :num_rand]

                return RoAffine(raffine, affine, other.model)
            elif self.model.mtype in 'SM' and other.model.mtype in 'VR':
                if self.model.top is not other.model.top:
                    raise ValueError('Models of operands mismatch.')
                affine = self.const @ other
                other = other.to_affine()
                num_rand = self.model.vars[-1].last

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
                raffine = affine_temp @ self.linear[:, :num_rand]

                roaffine = RoAffine(raffine, affine, self.model)

                if isinstance(other, DecAffine):
                    if not other.fixed:
                        msg = 'Affine decision rule '
                        msg += 'cannot be multiplied by random variables.'
                        raise TypeError(msg)
                    return DecRoAffine(roaffine, other.event_adapt, 'R')
                else:
                    return roaffine
        else:
            other = check_numeric(other)

            new_const = self.const @ other
            new_linear = sp_lmatmul(other, self, new_const.shape) @ self.linear

            return Affine(self.model, new_linear, new_const)

    def __rmatmul__(self, other):

        other = check_numeric(other)

        new_const = other @ self.const
        new_linear = sp_matmul(other, self, new_const.shape) @ self.linear

        return Affine(self.model, new_linear, new_const)

    def __add__(self, other):

        if isinstance(other, (Vars, VarSub, Affine)):
            other = other.to_affine()

            if self.model.mtype != other.model.mtype:
                if self.model.top is not other.model.top:
                    raise ValueError('Models of operands mismatch.')
                if self.model.mtype in 'VR':
                    temp = other.rand_to_roaffine(self.model)
                    return temp.__add__(self)
                elif other.model.mtype in 'VR':
                    temp = self.rand_to_roaffine(other.model)
                    return other.__add__(temp)
                else:
                    raise ValueError('Models of operands mismatch.')

            if self.model is not other.model:
                raise ValueError('Models of operands mismatch.')

            new_const = other.const + self.const

            if self.shape == other.shape:
                new_linear = add_linear(self.linear, other.linear)
            else:

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
                    new_linear = (self*np.ones(other.shape)).linear
                else:
                    new_linear = self.linear
        elif isinstance(other, Real):
            other = check_numeric(other)
            new_const = other + self.const
            new_linear = self.linear
        else:
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
        if isinstance(left, Affine) and not isinstance(left, DecAffine):
            return LinConstr(left.model, left.linear,
                             -left.const.reshape((left.const.size, )),
                             np.zeros(left.const.size))
        else:
            return left.__le__(0)

    def __ge__(self, other):

        left = other - self
        if isinstance(left, Affine) and not isinstance(left, DecAffine):
            return LinConstr(left.model, left.linear,
                             -left.const.reshape((left.const.size,)),
                             np.zeros(left.const.size))
        else:
            return left.__le__(0)

    def __eq__(self, other):

        left = self - other
        if isinstance(left, Affine) and not isinstance(left, DecAffine):
            return LinConstr(left.model, left.linear,
                             -left.const.reshape((left.const.size,)),
                             np.ones(left.const.size))
        else:
            return left.__eq__(0)


class Convex:
    """
    The Convex class creates an object of convex functions
    """

    __array_priority__ = 101

    def __init__(self, affine_in, affine_out, xtype, sign,
                 multiplier=1, sum_axis=False):

        self.model = affine_in.model
        self.affine_in = affine_in
        self.affine_out = affine_out
        self.multiplier = multiplier
        self.sum_axis = sum_axis
        self.size = affine_out.size
        self.xtype = xtype
        self.sign = sign

    def __repr__(self):
        xtypes = {'A': 'absolute expression',
                  'M': 'one-norm expression',
                  'E': 'Eclidean norm expression',
                  'I': 'infinity norm expression',
                  'S': 'element-wise square expression',
                  'Q': 'sum of squares expression',
                  'X': 'natural exponential expression',
                  'L': 'natural logarithm expression',
                  'P': 'entropy expression',
                  'K': 'KL divergence expression'}
        if self.affine_out.shape == ():
            shapes = 'an' if self.xtype in 'AEISP' else 'a'
        else:
            shapes = 'x'.join([str(dim) for dim in self.affine_out.shape])

        suffix = 's' if self.size > 1 else ''
        string = shapes + ' ' + xtypes[self.xtype] + suffix

        return string

    def __neg__(self):

        return Convex(self.affine_in, -self.affine_out, self.xtype, -self.sign,
                      self.multiplier)

    def __add__(self, other):

        cond1 = not isinstance(other, (Real, np.ndarray, Vars, Affine))
        cond2 = not sp.issparse(other)
        if cond1 and cond2:
            raise TypeError('The expression is not supported.')

        affine_in = self.affine_in
        affine_out = self.affine_out + other
        if self.xtype in 'S':
            affine_in = (affine_in.reshape(self.affine_out.shape) + 0*other)
            affine_in = affine_in.reshape(affine_in.size)
        if not isinstance(affine_out,
                          (Vars, VarSub, Affine, Real, np.ndarray)):
            raise TypeError('Incorrect data types.')

        new_convex = Convex(affine_in, affine_out,
                            self.xtype, self.sign, self.multiplier)

        return new_convex

    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):

        return self.__add__(-other)

    def __rsub__(self, other):

        return (-self).__add__(other)

    def __mul__(self, other):

        if not isinstance(other, Real):
            raise TypeError('Incorrect syntax.')

        if self.xtype in 'AMIEXLPK':
            multiplier = self.multiplier * abs(other)
        elif self.xtype in 'SQ':
            multiplier = self.multiplier * abs(other) ** 0.5
        else:
            raise ValueError('Unknown type of convex function.')

        return Convex(self.affine_in, other * self.affine_out,
                      self.xtype, np.sign(other)*self.sign, multiplier)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __le__(self, other):

        left = self - other
        if left.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return CvxConstr(left.model, left.affine_in, left.affine_out,
                         left.multiplier, left.xtype)

    def __ge__(self, other):

        right = other - self
        if right.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return CvxConstr(right.model, right.affine_in, right.affine_out,
                         right.multiplier, right.xtype)

    def __eq__(self, other):

        raise TypeError('Convex expressions are not applied to equality constraints')
    
    def sum(self, axis=None):

        if self.xtype not in 'XL':
            raise ValueError('The convex function does not support the sum() method.')

        return Convex(self.affine_in, self.affine_out.sum(axis=axis),
                      self.xtype, self.sign, self.multiplier, axis)


class PerspConvex(Convex):

    def __init__(self, affine_in, affine_scale, affine_out, xtype, sign, multiplier=1):

        super().__init__(affine_in, affine_out, xtype, sign, multiplier)
        self.affine_scale = affine_scale
    
    def __repr__(self):

        xtypes = {'X': 'natural exponential',
                  'L': 'natural logarithm'}

        if self.affine_out.shape == ():
            shapes = 'an' if self.xtype in 'AEISP' else 'a'
        else:
            shapes = 'x'.join([str(dim) for dim in self.affine_out.shape])

        suffix = 's' if self.size > 1 else ''
        # string = shapes + ' ' + 'perspective' + suffix  xtypes[self.xtype] + suffix
        string = f"{shapes} perspective expression{suffix} of the {xtypes[self.xtype]}"
        return string
    
    def __neg__(self):

        return PerspConvex(self.affine_in, self.affine_scale, -self.affine_out,
                           self.xtype, -self.sign, self.multiplier)

    def __add__(self, other):

        convex = super().__add__(other)
        
        return PerspConvex(convex.affine_in, self.affine_scale, convex.affine_out,
                           convex.xtype, convex.sign, convex.multiplier)
    
    def __radd__(self, other):

        return self.__add__(other)
    
    def __mul__(self, other):

        convex = super().__mul__(other)
        
        return PerspConvex(convex.affine_in, self.affine_scale, convex.affine_out,
                           convex.xtype, convex.sign, convex.multiplier)
    
    def __rmul__(self, other):

        return self.__mul__(other)
    
    def __le__(self, other):

        left = self - other
        if left.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return PCvxConstr(left.model, 
                          left.affine_in, left.affine_scale, left.affine_out,
                          left.multiplier, left.xtype)
    
    def __ge__(self, other):

        right = other - self
        if right.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return PCvxConstr(right.model, 
                          right.affine_in, right.affine_scale, right.affine_out,
                          right.multiplier, right.xtype)


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

    def __repr__(self):

        if self.shape == ():
            string = 'a '
        else:
            string = 'x'.join([str(dim) for dim in self.shape]) + ' '
        suffix = 's' if self.size > 1 else ''
        string += 'bi-affine expression' + suffix

        return string

    def __getitem__(self, item):

        if self.affine.sparray is None:
            self.affine.sparray = self.affine.sv_array()

        indices = self.affine.sparray[item]

        trans_array = sv_to_csr(indices)
        raffine = trans_array @ self.raffine
        affine = self.affine[item]

        return RoAffine(raffine, affine, self.rand_model)

    def reshape(self, shape):

        return RoAffine(self.raffine, self.affine.reshape(shape),
                        self.rand_model)

    @property
    def T(self):

        raffine = sp_trans(self) @ self.raffine
        affine = self.affine.T

        return RoAffine(raffine, affine, self.rand_model)

    def __neg__(self):

        return RoAffine(-self.raffine, -self.affine, self.rand_model)

    def __add__(self, other):

        if isinstance(other, (DecRule, DecRuleSub)):
            return self.__add__(other.to_affine())
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
                other = np.array([other]).reshape(())

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

            affine = self.affine + other
            return RoAffine(raffine, affine, self.rand_model)
        else:
            raise TypeError('Expression not supported.')

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

        new_raffine = sparse_mul(other, self) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __rmul__(self, other):

        new_affine = other * self.affine
        if isinstance(other, Real):
            other = np.array([other])

        new_raffine = sparse_mul(other, self) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __matmul__(self, other):

        other = check_numeric(other)

        new_affine = self.affine @ other

        new_raffine = sp_lmatmul(other, self, new_affine.shape) @ self.raffine

        return RoAffine(new_raffine, new_affine, self.rand_model)

    def __rmatmul__(self, other):

        other = check_numeric(other)

        new_affine = other @ self.affine

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

    def __eq__(self, other):

        left = self - other
        return RoConstr(left, sense=1)


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

    def __repr__(self):

        size = self.linear.shape[0]
        if size == 1:
            return '1 linear constraint'
        else:
            return '{} linear constraints'.format(size)


class CvxConstr:
    """
    The CvxConstr class creates an object of convex constraints
    """

    def __init__(self, model, affine_in, affine_out, multiplier, xtype):

        self.model = model
        self.affine_in = affine_in
        self.affine_out = affine_out
        self.multiplier = multiplier
        self.xtype = xtype

    def __repr__(self):

        size = self.affine_out.size
        if size == 1:
            return '1 convex constraint'
        else:
            return '{} convex constraints'.format(size)


class PCvxConstr(CvxConstr):

    def __init__(self, model, affine_in, affine_scale, affine_out, 
                 multiplier, xtype):
        
        super().__init__(model, affine_in, affine_out, multiplier, xtype)
        self.affine_scale = affine_scale


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
    """
    The ConeConstr class creates an object of second-order cone constraints
    """

    def __init__(self, model, left_var, left_index, right_var, right_index):

        self.model = model
        self.left_var = left_var
        self.right_var = right_var
        self.left_index = left_index
        self.right_index = right_index


class ExpConstr:
    """
    The ExpConstr class creates an object of exponential cone constraints
    """

    def __init__(self, model, expr1, expr2, expr3):
        self.model = model
        self.expr1 = expr1
        self.expr2 = expr2
        self.expr3 = expr3

    def __repr__(self):

        if isinstance(self.expr2, Real):
            size = 1
        else:
            size = self.expr2.size
        if size == 1:
            return '1 exponential conic constraint'
        else:
            return '{} exponential conic constraints'.format(size)


class KLConstr:
    """
    The KLConstr class creates an object of constraint for KL divergence
    """

    def __init__(self, p, phat, r):
        self.model = p.model
        self.p = p
        self.phat = phat
        self.r = r

    def __repr__(self):

        ns = self.p.size
        suffix = 's' if ns > 1 else ''

        return "KL divergence constraint for {} scenario{}".format(ns, suffix)


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

    def __repr__(self):

        size = self.affine.size
        if size == 1:
            return '1 robust constraint'
        else:
            return '{} robust constraints'.format(size)

    def forall(self, *args):
        """
        Specify the uncertainty set of the constraints involving random
        variables. The given arguments are constraints or collections of
        constraints used for defining the uncertainty set.

        Notes
        -----
        The uncertainty set defined by this method overrides the default
        uncertainty set defined for the worst-case objective.
        """

        constraints = []
        for items in args:
            if isinstance(items, Iterable):
                constraints.extend(list(items))
            else:
                constraints.append(items)

        sup_model = self.rand_model
        sup_model.reset()
        for item in constraints:
            if item.model is not sup_model:
                raise ValueError('Models mismatch.')
            sup_model.st(item)

        self.support = sup_model.do_math(primal=False, obj=False)

        return self

    def le_to_rc(self, support=None):

        num_constr, num_rand = self.raffine.shape
        support = self.support if not support else support
        if support is None:
            raise RuntimeError('The support of random variables is undefined.')
        size_support = support.linear.shape[1]
        num_rand = min(num_rand, support.linear.shape[0])

        dual_var = self.dec_model.dvar((num_constr, size_support))

        constr1 = (dual_var@support.obj +
                   self.affine.reshape(num_constr) <= 0)

        left = dual_var @ support.linear[:num_rand].T
        left = left + self.raffine[:, :num_rand] * support.const[:num_rand]
        sense2 = np.tile(support.sense[:num_rand], num_constr)
        num_rc_constr = left.const.size
        constr2 = LinConstr(left.model, left.linear,
                            -left.const.reshape(num_rc_constr),
                            sense2)

        bounds: List[LinConstr] = []
        index_pos = (support.ub == 0)
        if any(index_pos):
            bounds.append(dual_var[:, index_pos] <= 0)
        index_neg = (support.lb == 0)
        if any(index_neg):
            bounds.append(dual_var[:, index_neg] >= 0)

        if num_rand == support.linear.shape[0]:
            constr_list = [constr1, constr2]
            constr_list += [] if bounds is None else bounds
            # constr_list = ((constr1, constr2) if bounds is None else
            #                     (constr1, constr2, bounds))
        else:
            left = dual_var @ support.linear[num_rand:].T
            sense3 = np.tile(support.sense[num_rand:], num_constr)
            num_rc_constr = left.const.size
            constr3 = LinConstr(left.model, left.linear,
                                left.const.reshape(num_rc_constr),
                                sense3)
            constr_list = [constr1, constr2, constr3]
            constr_list += [] if bounds is None else bounds

        for n in range(num_constr):
            for qconstr in support.qmat:
                indices = np.array(qconstr, dtype=int) + n*size_support
                cone_constr = ConeConstr(self.dec_model, dual_var, indices[1:],
                                         dual_var, indices[0])
                constr_list.append(cone_constr)
            for xconstr in support.xmat:
                indices = xconstr
                cone_constr = ExpConstr(self.dec_model,
                                        dual_var[n, indices[0]],
                                        dual_var[n, indices[1]],
                                        dual_var[n, indices[2]])
                constr_list.append(cone_constr)

        return constr_list


class DecVar(Vars):
    """
    The DecVar class creates an object of generic variable array
    (here-and-now or wait-and-see) for adaptive DRO models
    """

    def __init__(self, dro_model, dvars, fixed=True, name=None):

        super().__init__(dvars.model, dvars.first, dvars.shape,
                         dvars.vtype, dvars.name)
        self.dro_model = dro_model
        self.event_adapt: List[List[int]] = [list(range(dro_model.num_scen))]
        self.rand_adapt = None
        self.ro_first = - 1
        self.fixed = fixed
        self.name = name

    def __repr__(self):

        var_name = '' if not self.name else self.name + ': '
        string = var_name
        expr = 'event-wise ' if len(self.event_adapt) > 1 else ''
        expr += 'static ' if self.fixed else 'affinely adaptive '

        if self.shape == ():
            string += 'an ' if expr[0] in 'ea' else 'a '
        else:
            string += 'x'.join([str(size) for size in self.shape]) + ' '

        suffix = 's' if self.size > 1 else ''

        string += expr + 'decision variable' + suffix

        return string

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]
        # if not isinstance(indices, np.ndarray):
        #     indices = np.array([indices]).reshape((1, ) * self.ndim)

        return DecVarSub(self.dro_model, self, indices, fixed=self.fixed)

    def to_affine(self):

        expr = super().to_affine()
        return DecAffine(self.dro_model, expr, self.event_adapt, self.fixed)

    def adapt(self, to):

        if isinstance(to, (Scen, Sized, int)):
            self.evtadapt(to)
        elif isinstance(to, (RandVar, RandVarSub)):
            self.affadapt(to)
        else:
            raise TypeError('Can not define adaption for the inputs.')

    def evtadapt(self, scens):

        if isinstance(scens, Scen):
            events = scens.series
        else:
            events = scens
        # events = list(events) if isinstance(events, Iterable) else [events]
        events = [events] if isinstance(events, (str, Real)) else list(events)

        for event in events:
            index = self.dro_model.series_scen[event]
            if index in self.event_adapt[0]:
                self.event_adapt[0].remove(index)
            else:
                raise KeyError('Wrong scenario index or {0} '.format(event) +
                               'has been redefined.')

        if not self.event_adapt[0]:
            self.event_adapt.pop(0)

        self.event_adapt.append(list(self.dro_model.series_scen[events]))

    def affadapt(self, rvars):

        self.fixed = False
        if self.shape == ():
            self.shape = (1, )
            self[:].affadapt(rvars)
            self.shape = ()
        else:
            self[:].affadapt(rvars)

    def __le__(self, other):

        return self.to_affine().__le__(other)

    def __ge__(self, other):

        return self.to_affine().__ge__(other)

    def __eq__(self, other):

        return self.to_affine().__eq__(other)

    def get(self, rvar=None):
        """
        Return the optimal solution of the decision variable, in
        terms of the optimal static decision or optimal coefficients
        of the decision rule.

        Parameters
        ----------
        rvar : RandVar
            The random varaible that the affine decision rule
            coefficients are with respect to

        Notes
        -----
        If the decision variable is event-wise, the method returns a
        series, where each element is a NumPy array representing the
        optimal solution for each scenario. If the decision is not
        adaptive to scenarios, the method returns a NumPy array.

        If the decision is static, the scenario-wise solution is a
        NumPy array with the same shape of the decision variable.

        If the decision is affinely adapt to random varaibles, the
        scenario-wise solution is 1) the constant term of the affine
        decision rule, if the argument rvar is unspecified; and 2) the
        linear ceofficients of the decision rule with respect to the
        random variable specified by rvar.

        An error message is raised if:
        1. The variable is not a decision variable.
        2. The model is unsolved, or no solution is obtained due to
        infeasibility, unboundedness, or numeric issues.
        """

        dro_model = self.dro_model
        if dro_model.solution is None:
            raise RuntimeError('The model is unsolved or infeasible')

        var_sol = dro_model.ro_model.rc_model.vars[1].get()
        # num_scen = dro_model.num_scen
        edict = event_dict(self.event_adapt)
        if rvar is None:
            outputs = []
            for eindex in range(len(self.event_adapt)):
                indices = (self.ro_first + eindex*self.size
                           + np.arange(self.size, dtype=int))
                result = var_sol[indices]
                if self.shape == ():
                    result = result[0]
                else:
                    result = result.reshape(self.shape)
                outputs.append(result)

            if len(outputs) > 1:
                ind_label = self.dro_model.series_scen.index
                return pd.Series([outputs[edict[key]] for key in edict],
                                 index=ind_label)
            else:
                return outputs[0]
        else:
            outputs = []
            drule_list = dro_model.rule_var()
            if isinstance(drule_list[0], Affine):
                raise ValueError('Decision not affinely adaptive!')
            for eindex in self.event_adapt:
                s = eindex[0]
                drule = drule_list[s]

                sp = drule.raffine[self.get_ind(), :][:, rvar.get_ind()].linear
                sp = coo_matrix(sp)

                sol_vec = np.array(dro_model.solution.x)[sp.col]
                sol_indices = sp.row

                # coeff = np.ones(sp.shape[0]) * np.NaN
                coeff = np.array([np.NaN] * sp.shape[0])
                coeff[sol_indices] = sol_vec

                # if rvar.to_affine().size == 1:
                #     outputs.append(coeff.reshape(self.shape))
                # else:
                rv_shape = rvar.to_affine().shape
                outputs.append(coeff.reshape(self.shape + rv_shape))

            if len(outputs) > 1:
                ind_label = self.dro_model.series_scen.index
                return pd.Series([outputs[edict[key]] for key in edict],
                                 index=ind_label)
            else:
                return outputs[0]

    @property
    def E(self):

        return DecAffine(self.dro_model, self.to_affine(),
                         fixed=self.fixed, ctype='E')


class DecVarSub(VarSub):

    def __init__(self, dro_model, dvars, indices, fixed=True):

        super().__init__(dvars, indices)
        self.dro_model = dro_model
        self.event_adapt = dvars.event_adapt
        self.rand_adapt = dvars.rand_adapt
        self.dvars = dvars
        self.fixed = fixed

    def __repr__(self):

        var_name = '' if not self.name else 'slice of {}: '.format(self.name)
        string = var_name
        expr = 'event-wise ' if len(self.event_adapt) > 1 else ''
        expr += 'static ' if self.fixed else 'affinely adaptive '

        if isinstance(self.indices, np.ndarray):
            string += 'x'.join([str(dim) for dim in self.indices.shape]) + ' '
            size = np.prod(self.indices.shape)
        else:
            string += 'an ' if expr[0] in 'ea' else 'a '
            size = 1

        suffix = 's' if size > 1 else ''
        string += expr + 'decision variable' + suffix

        return string

    def to_affine(self):

        expr = super().to_affine()
        return DecAffine(self.dro_model, expr, self.event_adapt, self.fixed)

    def adapt(self, rvars):

        self.fixed = False
        if not isinstance(rvars, (RandVar, RandVarSub)):
            raise TypeError('Affine adaptation requires a random variable.')

        self.affadapt(rvars)

    def affadapt(self, rvars):

        if self.vtype in ['B', 'I']:
            raise ValueError('No affine adaptation for integer variables.')
        if self.dro_model is not rvars.model.top:
            raise ValueError('Model mismatch.')

        self.fixed = False
        if self.rand_adapt is None:
            sup_model = self.dro_model.sup_model
            self.rand_adapt = np.zeros((self.size, sup_model.vars[-1].last),
                                       dtype=np.int8)

        dec_indices = self.indices
        dec_indices = dec_indices.reshape((dec_indices.size, 1))
        rand_indices = rvars.get_ind()
        rand_indices = rand_indices.reshape(rand_indices.size)

        dec_indices_flat = (dec_indices *
                            np.ones(rand_indices.shape, dtype=int)).flatten()
        rand_indices_flat = (np.ones(dec_indices.shape, dtype=int) *
                             rand_indices).flatten()

        if self.rand_adapt[dec_indices_flat, rand_indices_flat].any():
            raise RuntimeError('Redefinition of adaptation is not allowed.')

        self.rand_adapt[dec_indices_flat, rand_indices_flat] = 1
        self.dvars.rand_adapt = self.rand_adapt

    def __le__(self, other):

        return self.to_affine().__le__(other)

    def __ge__(self, other):

        return self.to_affine().__ge__(other)

    def __eq__(self, other):

        return self.to_affine().__eq__(other)

    @property
    def E(self):

        return DecAffine(self.dro_model, self.to_affine(),
                         fixed=self.fixed, ctype='E')


class RandVar(Vars):
    """
    The RandVar class creates an object of random variable array
    """

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

        return RandVarSub(self, indices)

    def __add__(self, other):

        expr = super().__add__(other)
        if isinstance(expr, RoAffine):
            expr = DecRoAffine(expr, other.event_adapt, other.ctype)

        return expr

    def __neg__(self):

        return super().__neg__()

    def __mul__(self, other):

        return super().__mul__(other)

    def __matmul__(self, other):

        return super().__matmul__(other)

    def __rmatmul__(self, other):

        return super().__rmatmul__(other)


class RandVarSub(VarSub):

    def __init__(self, rvars, indices):

        super().__init__(rvars, indices)
        self.e = VarSub(rvars.e, indices)

    @property
    def E(self):

        return self.e


class DecAffine(Affine):

    def __init__(self, dro_model, affine,
                 event_adapt=None, fixed=True, ctype='R'):

        super().__init__(affine.model, affine.linear,
                         affine.const, affine.sparray)
        self.dro_model = dro_model
        self.event_adapt = (event_adapt if event_adapt else
                            [list(range(dro_model.num_scen))])
        self.fixed = fixed
        self.ctype = ctype

    def __repr__(self):

        string = 'worst-case expectation of ' if self.ctype == 'E' else ''
        suffix = 's' if self.size > 1 else ''

        expr = 'affine expression' if self.fixed else 'bi-affine expression'
        event = 'event-wise ' if len(self.event_adapt) > 1 else ''

        if self.shape == ():
            string += 'an ' if (event + expr)[0] in 'ea' else 'a '
        else:
            string += 'x'.join([str(dim) for dim in self.shape]) + ' '

        string += event + expr + suffix

        return string

    def reshape(self, shape):

        expr = super().reshape(shape)

        return DecAffine(self.dro_model, expr,
                         self.event_adapt, self.fixed, self.ctype)

    def to_affine(self):

        expr = super().to_affine()

        return DecAffine(self.dro_model, expr, self.event_adapt,
                         self.fixed, self.ctype)

    @property
    def T(self):

        expr = super().T

        return DecAffine(self.dro_model, expr, self.event_adapt,
                         self.fixed, self.ctype)

    def __getitem__(self, indices):

        expr = super().__getitem__(indices)

        return DecAffine(self.dro_model, expr, self.event_adapt,
                         self.fixed, self.ctype)

    def __mul__(self, other):

        expr = super().__mul__(other)
        if isinstance(expr, Affine):
            return DecAffine(self.dro_model, expr,
                             event_adapt=self.event_adapt,
                             ctype=self.ctype, fixed=self.fixed)
        elif isinstance(expr, RoAffine):
            if not self.fixed:
                msg = 'Affine decision rule '
                msg += 'cannot be multiplied by random variables.'
                raise TypeError(msg)
            return DecRoAffine(expr, self.event_adapt, 'R')

    def __rmul__(self, other):

        return self.__mul__(other)

    def __matmul__(self, other):

        expr = super().__matmul__(other)
        if isinstance(expr, Affine):
            return DecAffine(self.dro_model, expr,
                             event_adapt=self.event_adapt,
                             ctype=self.ctype, fixed=self.fixed)
        elif isinstance(expr, RoAffine):
            if not self.fixed:
                msg = 'Affine decision rule '
                msg += 'cannot be multiplied by random variables.'
                raise TypeError(msg)
            return DecRoAffine(expr, self.event_adapt, 'R')

    def __rmatmul__(self, other):

        if isinstance(other, (Real, np.ndarray)) or sp.issparse(other):
            expr = super().__rmatmul__(other)
        else:
            expr = other.__matmul__(super().to_affine())

        if isinstance(expr, Affine):
            return DecAffine(self.dro_model, expr,
                             event_adapt=self.event_adapt,
                             ctype=self.ctype, fixed=self.fixed)
        elif isinstance(expr, RoAffine):
            if not self.fixed:
                msg = 'Affine decision rule '
                msg += 'cannot be multiplied by random variables.'
                raise TypeError(msg)
            return DecRoAffine(expr, self.event_adapt, 'R')

    def __neg__(self):

        expr = super().__neg__()

        return DecAffine(self.dro_model, expr,
                         event_adapt=self.event_adapt,
                         ctype=self.ctype, fixed=self.fixed)

    def __add__(self, other):
        
        expr = super().__add__(other)
        
        if isinstance(other, (DecAffine, DecVar, DecVarSub)):
            other = other.to_affine()
            self_is_fixed = self.fixed and len(self.event_adapt) == 1
            other_is_fixed = other.fixed and len(other.event_adapt) == 1
            if self.ctype == 'E' and other.ctype == 'R' and not other_is_fixed:
                raise TypeError('Incorrect expectation expressions.')
            if other.ctype == 'E' and self.ctype == 'R' and not self_is_fixed:
                raise TypeError('Incorrect expectation expressions.')
            event_adapt = comb_set(self.event_adapt, other.event_adapt)
            ctype = other.ctype
            fixed = other.fixed
        elif isinstance(other, DecRoAffine):
            event_adapt = comb_set(self.event_adapt, other.event_adapt)
            ctype = other.ctype
            fixed = False
        elif isinstance(other, (Real, np.ndarray, Affine, RoAffine,
                                Vars, VarSub)) or sp.issparse(other):
            event_adapt = self.event_adapt
            ctype = 'R'
            fixed = True
        else:
            return other.__add__(self)

        fixed = fixed and self.fixed
        ctype = 'E' if 'E' in (self.ctype + ctype) else 'R'

        if isinstance(expr, Affine):
            return DecAffine(self.dro_model, expr,
                             event_adapt=event_adapt,
                             ctype=ctype, fixed=fixed)
        elif isinstance(expr, RoAffine):
            if isinstance(other, DecRoAffine):
                ctype = other.ctype
            else:
                ctype = 'R'
            return DecRoAffine(expr, event_adapt, ctype)

    def __abs__(self):

        expr = super().__abs__()

        return DecConvex(expr, self.event_adapt)

    def abs(self):

        return self.__abs__()

    def norm(self, degree):
        """
        Return the first, second, or infinity norm of a 1-D array.

        Refer to `rsome.norm` for full documentation

        See Also
        --------
        rso.norm : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().norm(degree)

        return DecConvex(expr, self.event_adapt)

    def square(self):
        """
        Return the element-wise square of an array.

        Refer to `rsome.square` for full documentation

        See Also
        --------
        rso.square : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().square()

        return DecConvex(expr, self.event_adapt)

    def sumsqr(self):
        """
        Return the sum of squares of a 1-D array.

        Refer to `rsome.sumsqr` for full documentation.

        See Also
        --------
        rso.sumsqr : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().sumsqr()

        return DecConvex(expr, self.event_adapt)

    def sum(self, axis=None):

        expr = super().sum(axis)

        return DecAffine(self.dro_model, expr, self.event_adapt, self.fixed)

    def expcone(self, x, z):
        """
        Return the exponential cone constraint z*exp(x/z) <= affine

        Refer to `rsome.expcone` for full documentation.

        See Also
        --------
        rso.expcone : equivalent function
        """

        event_adapt = self.event_adapt

        if isinstance(x, (DecVar, DecVarSub)):
            if x.to_affine().size > 1:
                raise ValueError('The expression of x must be a scalar.')
        elif isinstance(x, (DecAffine, np.ndarray)):
            if x.size > 1:
                raise ValueError('The expression of x must be a scalar')

        if isinstance(x, (DecVar, DecVarSub, DecAffine)):
            event_adapt = comb_set(event_adapt, x.event_adapt)

        if isinstance(z, (DecVar, DecVarSub)):
            if z.to_affine().size > 1:
                raise ValueError('The expression of z must be a scalar.')
        elif isinstance(z, (DecAffine, np.ndarray)):
            if z.size > 1:
                raise ValueError('The expression of z must be a scalar')

        if isinstance(z, (DecVar, DecVarSub, DecAffine)):
            event_adapt = comb_set(event_adapt, z.event_adapt)

        return DecExpConstr(ExpConstr(self.model, x, self, z), event_adapt)

    def exp(self):
        """
        Return the natural exponential function exp(affine)

        Refer to `rsome.exp` for full documentation.

        See Also
        --------
        rso.exp : equivalent function
        """

        # if self.size > 1:
        #     raise ValueError('The expression must be a scalar')

        return DecConvex(Convex(self, np.zeros(self.shape), 'X', 1), 
                         self.event_adapt)
    
    def pexp(self, scale):
        """
        Return the perspective of natural exponential function 
        scale * exp(affine/scale)

        Refer to `rsome.pexp` for full documentation.

        See Also
        --------
        rso.pexp : equivalent function
        """

        # if self.size > 1:
        #     raise ValueError('The expression must be a scalar')

        return DecPerspConvex(PerspConvex(self, scale, np.zeros(self.shape), 'X', 1), 
                              self.event_adapt)

    def log(self):
        """
        Return the natural logarithm function log(affine)

        Refer to `rsome.log` for full documentation.

        See Also
        --------
        rso.log : equivalent function
        """

        # if self.size > 1:
        #     raise ValueError('The expression must be a scalar')

        return DecConvex(Convex(self, np.zeros(self.shape), 'L', -1), 
                         self.event_adapt)
    
    def plog(self, scale):
        """
        Return the perspective of natural logarithm function
        scale * log(affine/scale)

        Refer to `rsome.plog` for full documentation.

        See Also
        --------
        rso.plog : equivalent function
        """

        # if self.size > 1:
        #     raise ValueError('The expression must be a scalar')

        return DecPerspConvex(PerspConvex(self, scale, np.zeros(self.shape), 'L', -1),
                              self.event_adapt)

    def entropy(self):
        """
        Return the natural exponential function -sum(affine*log(affine))

        Refer to `rsome.entropy` for full documentation.

        See Also
        --------
        rso.entropy : equivalent function
        """

        # if self.size > 1:
        #     raise ValueError('The expression must be a scalar')
        if self.shape != ():
            if self.size != max(self.shape):
                raise ValueError('The expression must be a vector.')

        return DecConvex(Convex(self, np.float64(0), 'P', -1), self.event_adapt)

    def __le__(self, other):

        left = self - other

        if isinstance(left, DecAffine):
            return DecLinConstr(left.model, left.linear, -left.const,
                                np.zeros(left.size), left.event_adapt,
                                left.fixed, left.ctype)
        elif isinstance(left, DecRoAffine):
            return DecRoConstr(left, 0, left.event_adapt, left.ctype)
        elif isinstance(left, DecConvex):
            if left.sign == -1:
                raise ValueError('Nonconvex constraints.')
            return DecCvxConstr(left, left.event_adapt)
        elif isinstance(left, DecPerspConvex):
            if left.sign == -1:
                raise ValueError('Nonconvex constraints.')
            constr = PCvxConstr(left.model, 
                                left.affine_in, left.affine_scale, left.affine_out,
                                left.multiplier, left.xtype)
            return DecPCvxConstr(constr, left.event_adapt)

    def __ge__(self, other):

        left = other - self

        if isinstance(left, DecAffine):
            return DecLinConstr(left.model, left.linear, -left.const,
                                np.zeros(left.size), left.event_adapt,
                                left.fixed, left.ctype)
        elif isinstance(left, DecRoAffine):
            return DecRoConstr(left, 0, left.event_adapt, left.ctype)
        elif isinstance(left, DecConvex):
            if left.sign == -1:
                raise ValueError('Nonconvex constraints.')
            return DecCvxConstr(left, left.event_adapt)
        elif isinstance(left, DecPerspConvex):
            if left.sign == -1:
                raise ValueError('Nonconvex constraints.')
            constr = PCvxConstr(left.model, 
                                left.affine_in, left.affine_scale, left.affine_out,
                                left.multiplier, left.xtype)
            return DecPCvxConstr(constr, left.event_adapt)

    def __eq__(self, other):

        left = self - other
        if isinstance(left, DecAffine):
            return DecLinConstr(left.model, left.linear, -left.const,
                                np.ones(left.size), left.event_adapt,
                                left.fixed, left.ctype)
        elif isinstance(left, DecRoAffine):
            return DecRoConstr(left, 1, left.event_adapt, left.ctype)

    @property
    def E(self):

        affine = Affine(self.model, self.linear, self.const)
        return DecAffine(self.dro_model, affine, fixed=self.fixed, ctype='E')


class DecConvex(Convex):

    def __init__(self, convex, event_adapt):

        super().__init__(convex.affine_in, convex.affine_out,
                         convex.xtype, convex.sign, convex.multiplier)
        self.event_adapt = event_adapt

    def __neg__(self):

        expr = super().__neg__()
        return DecConvex(expr, self.event_adapt)

    def __add__(self, other):

        expr = super().__add__(other)

        if isinstance(other, (Real, np.ndarray)) or sp.issparse(other):
            event_adapt = self.event_adapt
        else:
            event_adapt = comb_set(self.event_adapt, other.event_adapt)

        return DecConvex(expr, event_adapt)

    def __mul__(self, other):

        expr = super().__mul__(other)

        return DecConvex(expr, self.event_adapt)

    def __rmul__(self, other):

        expr = super().__rmul__(other)

        return DecConvex(expr, self.event_adapt)

    def __le__(self, other):

        constr = super().__le__(other)

        return DecCvxConstr(constr, self.event_adapt)

    def __ge__(self, other):

        constr = super().__ge__(other)

        return DecCvxConstr(constr, self.event_adapt)


class DecPerspConvex(PerspConvex):

    def __init__(self, convex, event_adapt):

        super().__init__(convex.affine_in, convex.affine_scale, convex.affine_out,
                         convex.xtype, convex.sign, convex.multiplier)
        self.event_adapt = event_adapt
    
    def __neg__(self):

        expr = super().__neg__()
        return DecPerspConvex(expr, self.event_adapt)

    def __add__(self, other):

        expr = super().__add__(other)

        if isinstance(other, (Real, np.ndarray)) or sp.issparse(other):
            event_adapt = self.event_adapt
        else:
            event_adapt = comb_set(self.event_adapt, other.event_adapt)

        return DecPerspConvex(expr, event_adapt)

    def __mul__(self, other):

        expr = super().__mul__(other)

        return DecPerspConvex(expr, self.event_adapt)

    def __rmul__(self, other):

        expr = super().__rmul__(other)

        return DecPerspConvex(expr, self.event_adapt)

    def __le__(self, other):

        constr = super().__le__(other)

        return DecPCvxConstr(constr, self.event_adapt)

    def __ge__(self, other):

        constr = super().__ge__(other)

        return DecPCvxConstr(constr, self.event_adapt)


class DecRoAffine(RoAffine):

    def __init__(self, roaffine, event_adapt, ctype):

        super().__init__(roaffine.raffine, roaffine.affine,
                         roaffine.rand_model)

        self.event_adapt = event_adapt
        self.ctype = ctype

    def __repr__(self):

        event = 'event-wise ' if len(self.event_adapt) > 1 else ''
        if self.shape == ():
            string = 'a ' if event == '' else 'an '
        else:
            string = 'x'.join([str(dim) for dim in self.shape]) + ' '
        suffix = 's' if self.size > 1 else ''
        string += event + 'bi-affine expression' + suffix

        return string

    def sum(self, axis=None):

        expr = super().sum(axis)

        return DecRoAffine(expr, self.event_adapt, self.ctype)

    def __neg__(self):

        expr = super().__neg__()

        return DecRoAffine(expr, self.event_adapt, self.ctype)

    def __add__(self, other):

        if isinstance(other, (DecVar, DecVarSub, DecAffine, DecRoAffine)):
            if isinstance(other, DecRoAffine):
                if self.ctype != other.ctype:
                    raise TypeError('Incorrect expectation expressions.')
            else:
                other = other.to_affine()
                if self.ctype != other.ctype:
                    if ((self.ctype == 'E'
                         and (not other.fixed or len(other.event_adapt) > 1))
                            or other.ctype == 'E'):
                        raise TypeError('Incorrect expectation expressions.')
                other = other.to_affine()
            event_adapt = comb_set(self.event_adapt, other.event_adapt)
            ctype = 'E' if 'E' in self.ctype + other.ctype else 'R'
        elif (isinstance(other, (Real, np.ndarray, RoAffine))
              or sp.issparse(other)):
            event_adapt = self.event_adapt
            ctype = self.ctype
        elif isinstance(other, (Vars, VarSub, Affine)):
            if other.model.mtype != 'V':
                if self.ctype == 'E':
                    raise ValueError('Incorrect affine expressions.')
            event_adapt = self.event_adapt
            ctype = self.ctype
        else:
            raise TypeError('Unknown expression type.')

        expr = super().__add__(other)

        return DecRoAffine(expr, event_adapt, ctype)

    def __sub__(self, other):

        return self.__add__(-other)

    def __mul__(self, other):

        expr = super().__mul__(other)

        return DecRoAffine(expr, self.event_adapt, self.ctype)

    def __rmul__(self, other):

        expr = super().__rmul__(other)

        return DecRoAffine(expr, self.event_adapt, self.ctype)

    def __matmul__(self, other):

        expr = super().__matmul__(other)

        return DecRoAffine(expr, self.event_adapt, self.ctype)

    def __rmatmul__(self, other):

        expr = super().__rmatmul__(other)

        return DecRoAffine(expr, self.event_adapt, self.ctype)

    def __le__(self, other):

        left = self - other

        return DecRoConstr(left, 0, left.event_adapt, left.ctype)

    def __ge__(self, other):

        left = other - self

        return DecRoConstr(left, 0, left.event_adapt, left.ctype)

    def __eq__(self, other):

        left = self - other

        return DecRoConstr(left, 1, left.event_adapt, left.ctype)

    @property
    def E(self):

        roaffine = RoAffine(self.raffine, self.affine, self.rand_model)
        return DecRoAffine(roaffine, self.event_adapt, ctype='E')


class DecLinConstr(LinConstr):

    def __init__(self, model, linear, const, sense,
                 event_adapt=None, fixed=True, ctype='R'):

        super().__init__(model, linear, const, sense)
        self.event_adapt = event_adapt
        self.fixed = fixed
        self.ctype = ctype
        self.ambset = None

    def __repr__(self):

        size = self.linear.shape[0]
        suffix = 's' if size > 1 else ''
        if self.ctype == 'E':
            event = ' '
            ctype = ' of expectation' + suffix
        else:
            event = ' event-wise ' if len(self.event_adapt) > 1 else ' '
            ctype = ''
        expr = 'linear' if self.fixed else 'robust'

        return '{}{}{} constraint{}{}'.format(size, event, expr, suffix, ctype)


class DecBounds(Bounds):

    def __init__(self, bounds, event_adapt=None):

        super().__init__(bounds.model, bounds.indices, bounds.values,
                         bounds.btype)
        self.event_adapt = event_adapt


class DecCvxConstr(CvxConstr):

    def __init__(self, constr, event_adapt):

        super().__init__(constr.model, constr.affine_in,
                         constr.affine_out, constr.multiplier, constr.xtype)
        self.event_adapt = event_adapt

class DecPCvxConstr(PCvxConstr):

    def __init__(self, constr, event_adapt):

        super().__init__(constr.model, constr.affine_in, constr.affine_scale,
                         constr.affine_out, constr.multiplier, constr.xtype)
        self.event_adapt = event_adapt


class DecExpConstr(ExpConstr):

    def __init__(self, constr, event_adapt):

        super().__init__(constr.model,
                         constr.expr1, constr.expr2, constr.expr3)
        self.event_adapt = event_adapt


class DecRoConstr(RoConstr):

    def __init__(self, roaffine, sense, event_adapt, ctype):

        super().__init__(roaffine, sense)

        self.event_adapt = event_adapt
        self.ctype = ctype
        self.ambset = None

    def __repr__(self):

        size = self.affine.size
        suffix = 's' if size > 1 else ''
        if self.ctype == 'E':
            event = ' '
            ctype = ' of expectation' + suffix
        else:
            event = ' event-wise ' if len(self.event_adapt) > 1 else ' '
            ctype = ''

        string = '{}{}robust constraint{}{}'.format(size, event, suffix, ctype)
        return string

    def forall(self, ambset):

        if isinstance(ambset, (LinConstr, Bounds, CvxConstr, Iterable)):
            suppset = flat([ambset])
            for constr in suppset:
                if constr.model is not self.rand_model:
                    raise ValueError('Models mismatch.')
            self.ambset = suppset
            return self
        else:
            if self.dec_model.top is not ambset.model:
                raise ValueError('Models mismatch.')

            self.ambset = ambset
            return self


class DecRule:

    __array_priority__ = 102

    def __init__(self, model, shape=(), name=None,):

        self.model = model
        self.name = name
        self.fixed = model.dvar(shape, 'C')
        self.shape = self.fixed.shape
        self.size = int(np.prod(self.shape))
        self.depend = None
        self.roaffine = None
        self.var_coeff = None

    def __repr__(self):

        suffix = 's' if np.prod(self.shape) > 1 else ''

        string = '' if self.name is None else self.name + ': '
        if self.shape == ():
            string += 'a '
        else:
            string += 'x'.join([str(size) for size in self.shape]) + ' '
        string += 'decision rule variable' + suffix

        return string

    def reshape(self, shape):

        return self.to_affine().reshape(shape)

    def adapt(self, rvar, ldr_indices=None):

        if self.roaffine is not None:
            raise SyntaxError('Adaptation must be defined ' +
                              'before used in constraints')

        if self.model is not rvar.model.top:
            raise ValueError('Models mismatch.')

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
            raise RuntimeError('Redefinition of adaptation is not allowed.')

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

    @property
    def T(self):

        return self.to_affine().T

    def __getitem__(self, item):

        item_array = index_array(self.shape)
        indices = item_array[item]
        # if not isinstance(indices, np.ndarray):
        #     indices = np.array([indices]).reshape((1, ) * indices.ndim)

        return DecRuleSub(self, indices, item)

    def __neg__(self):

        return - self.to_affine()

    def __add__(self, other):

        return self.to_affine().__add__(other)

    def __radd__(self, other):

        return self.__add__(other)

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

    def __eq__(self, other):

        return (self - other).__eq__(0)

    def get(self, rvar=None):
        """
        Return the optimal coefficients of the affine decision rule.

        Parameters
        ----------
        rvar : Vars
            The random varaible that the affine decision rule
            coefficients are with respect to

        Notes
        -----
        If the decision variable is event-wise, the method returns a
        series, where each element is a NumPy array representing the
        optimal solution for each scenario. If the decision is not
        adaptive to scenarios, the method returns a NumPy array.

        If the decision is static, the scenario-wise solution is a
        NumPy array with the same shape of the decision variable.

        If the decision is affinely adapt to random varaibles, the
        scenario-wise solution is 1) the constant term of the affine
        decision rule, if the argument rvar is unspecified; and 2) the
        linear ceofficients of the decision rule with respect to the
        random variable specified by rvar.

        An error message is raised if:
        1. The variable is not a decision variable.
        2. The model is unsolved, or no solution is obtained due to
        infeasibility, unboundedness, or numeric issues.
        """

        if self.model.solution is None:
            raise RuntimeError('The model is unsolved or infeasible')

        if rvar is None:
            return self.fixed.get()
        else:
            if rvar.model.mtype != 'S':
                raise ValueError('The input is not a random variable.')
            ldr_row, ldr_col = self.size, self.model.rc_model.vars[-1].last
            ldr_coeff = np.array([[np.NaN] * ldr_col] * ldr_row)
            rand_ind = rvar.get_ind()
            row_ind, col_ind = np.where(self.depend == 1)
            ldr_coeff[row_ind, col_ind] = self.var_coeff.get()

            rv_shape = rvar.to_affine().shape
            return ldr_coeff[:, rand_ind].reshape(self.shape + rv_shape)


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

        if isinstance(roaffine, Affine):
            linear = roaffine.linear[self.indices]
            const = roaffine.const.flatten()[self.indices].reshape(self.shape)
            return Affine(roaffine.model, linear, const)
        else:
            raffine = roaffine.raffine[self.indices, :]
            affine = roaffine.affine[self.item]

            return RoAffine(raffine, affine, self.dec_rule.model.sup_model)

    @property
    def T(self):

        return self.to_affine().T

    def __neg__(self):

        return - self.to_affine()

    def __add__(self, other):

        return self.to_affine().__add__(other)

    def __radd__(self, other):

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

    def __eq__(self, other):

        return (self - other).__eq__(0)


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

    def __repr__(self):

        linear = self.linear
        nc, nb, ni = (sum(self.vtype == 'C'),
                      sum(self.vtype == 'B'),
                      sum(self.vtype == 'I'))
        nineq, neq = sum(self.sense == 0), sum(self.sense == 1)
        nnz = self.linear.indptr[-1]

        string = '=============================================\n'
        string += 'Number of variables:           {0}\n'.format(linear.shape[1])
        string += 'Continuous/binaries/integers:  {0}/{1}/{2}\n'.format(nc, nb, ni)
        string += '---------------------------------------------\n'
        string += 'Number of linear constraints:  {0}\n'.format(linear.shape[0])
        string += 'Inequalities/equalities:       {0}/{1}\n'.format(nineq, neq)
        string += 'Number of coefficients:        {0}\n'.format(nnz)

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

    def show(self):

        return self.showlc()

    def solve(self, solver):

        return solver.solve(self)

    def lp_export(self):

        string = 'Minimize\n'
        string += ' obj: '
        obj_str = ' '.join(['{} {} x{}'.format('-' if coeff < 0 else '+',
                                               abs(coeff), i+1)
                            for i, coeff in enumerate(self.obj[0]) if coeff])
        string += obj_str[2:] if obj_str[:2] == '+ ' else obj_str

        string += '\nSubject To\n'
        for i in range(self.linear.shape[0]):
            row = self.linear[i]
            coeffs = row.data
            indices = row.indices
            each = ['{} {} x{}'.format('-' if coeff < 0 else '+',
                                       abs(coeff), index+1)
                    for coeff, index in zip(coeffs, indices)]
            each_line = ' '.join(each)
            if each_line[:2] == '+ ':
                each_line = each_line[2:]

            string += ' c{}: '.format(i+1) + each_line
            string += ' <= ' if self.sense[i] == 0 else ' == '
            string += '{}\n'.format(self.const[i])

        ub, lb = self.ub, self.lb
        nvar = len(ub)
        ub_string = '\n'.join([' x{} <= {}'.format(i+1, ub[i])
                               for i in range(nvar) if ub[i] < np.inf])
        lb_string = '\n'.join([' {} <= x{}'.format(lb[i], i+1)
                               for i in range(nvar) if lb[i] > -np.inf])
        free_string = '\n'.join([' x{} free'.format(i+1)
                                 for i in range(nvar)
                                 if lb[i] == -np.inf and ub[i] == np.inf])
        string += 'Bounds\n'
        if len(ub_string) > 0:
            string += ub_string + '\n'
        if len(lb_string) > 0:
            string += lb_string + '\n'
        if len(free_string) > 0:
            string += free_string + '\n'

        ind_int, = np.where(self.vtype == 'I')
        int_string = '\n'.join(['x{}'.format(i+1) for i in ind_int])
        if len(ind_int) > 0:
            string += 'General\n'
            string += ' ' + int_string + '\n'

        ind_bin, = np.where(self.vtype == 'B')
        bin_string = '\n'.join(['x{}'.format(i+1) for i in ind_bin])
        if len(ind_bin) > 0:
            string += 'Binary\n'
            string += ' ' + bin_string + '\n'

        string += 'End'

        return string

    def to_lp(self, name='out'):
        '''
        Export the standard form of the optimization model as a .lp file.

        Parameters
        ----------
        name : file name of the .lp file

        Notes
        -----
        There is no need to specify the .lp extension. The default file name
        is "out".
        '''

        with open(name + '.lp', 'w') as f:
            f.write(self.lp_export())


class Solution:

    def __init__(self, objval, x, status, time):

        self.objval = objval
        self.x = x
        self.status = status
        self.time = time


class Scen:

    def __init__(self, ambset, series, pr):

        # super().__init__(data=series.values, index=series.index)
        self.ambset = ambset
        self.series = series
        self.p = pr

    def __repr__(self):

        if isinstance(self.series, Sized):
            return 'Scenario indices: \n' + self.series.__str__()
        else:
            return 'Scenario index: \n' + self.series.__str__()

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
        """
        Specify the support set(s) of an ambiguity set.

        Parameters
        ----------
        args : tuple
            Constraints or collections of constraints as iterable type of
            objects, used for defining the feasible region of the support set.

        Notes
        -----
        RSOME leaves the support set unspecified if the input argument is
        an empty iterable object.
        """

        args = flat(args)
        if len(args) == 0:
            return

        for arg in args:
            if arg.model is not self.ambset.model.sup_model:
                raise ValueError('Constraints are not for this support.')
            if not isinstance(arg, (LinConstr, CvxConstr, Bounds, ConeConstr)):
                raise TypeError('Invalid constraint type.')

        # for i in self.series:
        indices = (self.series if isinstance(self.series, pd.Series)
                   else [self.series])
        for i in indices:
            self.ambset.sup_constr[i] = tuple(args)

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

        args = flat(args)
        if len(args) == 0:
            return

        for arg in args:
            if arg.model is not self.ambset.model.exp_model:
                raise ValueError('Constraints are not defined for ' +
                                 'expectation sets.')

        self.ambset.exp_constr.append(tuple(args))
        indices: Iterable[int]
        if not isinstance(self.series, Iterable):
            indices = [self.series]
        else:
            indices = self.series
        self.ambset.exp_constr_indices.append(list(indices))


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

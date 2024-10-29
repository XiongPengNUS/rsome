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
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.linalg import sqrtm, eigh
from collections.abc import Iterable, Sized
# from typing import List


def def_sol(formula, display=True, log=False, params={}):
    """
    This is the default solver of RSOME.
    """

    try:
        if formula.qmat:
            warnings.warn('the LP solver ignores SOC constraints.')
    except AttributeError:
        pass

    try:
        if formula.xmat:
            warnings.warn('The LP solver ignores exponential cone constraints.')
        if formula.lmi:
            warnings.warn('The LP solver ignores semidefinite cone constraints.')
    except AttributeError:
        pass

    A = formula.linear
    sense = formula.sense
    vtype = formula.vtype
    num_constr = A.shape[0]

    if all(vtype == 'C'):
        indices_eq = (formula.sense == 1)
        indices_ineq = (formula.sense == 0)
        linear_eq = formula.linear[indices_eq, :] if len(indices_eq) else None
        linear_ineq = formula.linear[indices_ineq, :] if len(indices_ineq) else None
        const_eq = formula.const[indices_eq] if len(indices_eq) else None
        const_ineq = formula.const[indices_ineq] if len(indices_ineq) else None

        bounds = [(lb, ub) for lb, ub in zip(formula.lb, formula.ub)]

        default = {'maxiter': 1000000000}

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
            objval = formula.obj @ res.x

            pi = np.ones(num_constr) * np.nan
            upi = res['upper']['marginals']
            lpi = res['lower']['marginals']
            pi[indices_eq] = res['eqlin']['marginals']
            pi[indices_ineq] = res['ineqlin']['marginals']
            y = {'pi': pi, 'upi': upi, 'lpi': lpi}

            return Solution('SciPy', objval, res.x, res.status, stime, y=y)
        else:
            status = res.status
            msg = 'Fail to find the optimal solution, '
            reasons = ('iteration limit is reached.' if status == 1 else
                       'the problem appears to be infeasible.' if status == 2 else
                       'the problem appears to be unbounded.' if status == 3 else
                       'numerical difficulties encountered.')
            msg += 'because {}'.format(reasons)
            warnings.warn(msg)
            return Solution('Scipy', np.nan, None, status, stime)
    else:
        b_u = formula.const
        b_l = np.array([-np.inf] * A.shape[0])
        bool_eq = (sense == 1)
        b_l[bool_eq] = b_u[bool_eq]

        bool_bin = (vtype == 'B')
        lb = formula.lb
        ub = formula.ub
        lb[bool_bin] = 0
        ub[bool_bin] = 1

        integrality = np.zeros(A.shape[1])
        integrality[vtype != 'C'] = 1

        if display:
            print('Being solved by the default MILP solver...', flush=True)
            time.sleep(0.2)
        t0 = time.time()
        if all(vtype == 'C'):
            linear_ineq = A[sense == 0]
            const_ineq = formula.const[sense == 0]
            linear_eq = A[sense == 1]
            const_eq = formula.const[sense == 1]
            bounds = [(lb, ub) for lb, ub in zip(formula.lb, formula.ub)]
            default = {'maxiter': 1000000000}
            res = opt.linprog(formula.obj, A_ub=linear_ineq, b_ub=const_ineq,
                              A_eq=linear_eq, b_eq=const_eq,
                              bounds=bounds, options=default)
        else:
            res = opt.milp(formula.obj,
                           constraints=opt.LinearConstraint(A, b_l, b_u),
                           bounds=opt.Bounds(lb, ub),
                           integrality=integrality)
        stime = time.time() - t0
        if display:
            print('Solution status: {0}'.format(res.status))
            print('Running time: {0:0.4f}s'.format(stime))

        if res.status == 0:
            objval = formula.obj @ res.x
            return Solution('SciPy', objval, res.x, res.status, stime)
        else:
            status = res.status
            msg = 'Fail to find the optimal solution.'
            warnings.warn(msg)
            return Solution('Scipy', np.nan, None, status, stime)


def concat(iters, axis=0):
    """
    Join a sequence of arrays of affine expressions along an existing axis.

    Parameters
    ----------
    iters : array_like.
        A sequence of RSOME variables or affine expressions. The arrays must
        have the same shape, except in the dimension corresponding to `axis`
        (the first, by default).

    axis : int, optional
        The axis along which the arrays will be joined.  Default is 0.

    Returns
    -------
    out : Affine
        The concatenated array of affine expressions.
    """

    linear_each = []
    const_each = []
    idx_each = []
    count = 0
    model = None
    event_adapt = None
    fixed = True
    ctype = None

    for item in iters:
        if isinstance(item, (Real, np.ndarray)):
            continue
        if not isinstance(item, (Affine, Vars, VarSub)):
            raise TypeError('Unsupported data type for concatenation')
        if model is None:
            model = item.model
            num_var = model.last
        else:
            if model != item.model:
                raise ValueError('Model mismatch.')

    if model is None:
        return np.concatenate(iters, axis)

    for item in iters:
        if isinstance(item, (Real, np.ndarray)):
            item_value = np.array(item)
            item_size = item_value.size
            item_linear = csr_matrix(([], ([], [])), shape=(item_size, num_var))
            item = Affine(model, item_linear, item_value)
            if model.mtype == 'V':
                item = DecAffine(model, item, [list(range(item.model.top.num_scen))])
        if not isinstance(item, Affine):
            item = item.to_affine()
        if isinstance(item, DecAffine):
            if event_adapt is None:
                event_adapt = [list(range(item.model.top.num_scen))]
            if ctype is None:
                ctype = item.ctype
            if ctype != item.ctype:
                raise ValueError('Cannot concatenate different types of expressions.')
            event_adapt = comb_set(event_adapt, item.event_adapt)
            fixed = fixed and item.fixed

        if item.linear.shape[1] < num_var:
            item.linear.resize(item.linear.shape[0], num_var)
        linear_each.append(item.linear)
        const_each.append(item.const)
        idx_each.append(np.arange(count, count+item.size).reshape(item.shape))

        count += item.size

    ndim = max([i.ndim for i in idx_each])
    idx_each = [i.reshape([1] * ndim) if i.shape == () else i
                for i in idx_each]
    const_each = [const.reshape([1] * ndim) if const.shape == () else const
                  for const in const_each]

    idx_all = np.concatenate(idx_each, axis=axis).flatten()
    linear_all = sp.vstack(linear_each)[idx_all]
    const_all = np.concatenate(const_each, axis=axis)

    affine = Affine(item.model, linear_all, const_all)
    if event_adapt is None:
        return affine
    else:
        return DecAffine(item.model.top, affine, event_adapt, fixed, ctype)


def rstack(*args):
    """
    Stack a sequence of rows of affine expressions vertically (row wise).

    Parameters
    ----------
    arg : {list, Affine}.
        Each arg represents an array of affine expressions. If arg is a list
        of affine expressions, they will be concatenated horizontally (column
        wise) first.

    Returns
    -------
    out : Affine
        The vertically stacked array of affine expressions.

    Notes
    -----
    The rstack function is different from the vstack function from the numpy
    package in 1) the arrays to be stacked together are presented as separate
    arguments, instead of elements in an array-like sequence; and 2) a list of
    arrays can be stacked horizontally first before being vertically stacked.
    """

    rows = []
    for arg in args:
        if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
            rows.append(concat(arg, axis=1))
        else:
            rows.append(arg)

    return concat(rows, axis=0)


def cstack(*args):
    """
    Stack a sequence of rows of affine expressions horizontally (column wise).

    Parameters
    ----------
    arg : {list, Affine}.
        Each arg represents an array of affine expressions. If arg is a list of
        affine expressions, they will be concatenated vertically (row wise) first.

    Returns
    -------
    out : Affine
        The horizontally stacked array of affine expressions.

    Notes
    -----
    The cstack function is different from the hstack function from the numpy
    package in 1) the arrays to be stacked together are presented as separate
    arguments, instead of elements in an array-like sequence; and 2) a list of
    arrays can be stacked vertically first before being horizontally stacked.
    """

    cols = []
    for arg in args:
        if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
            cols.append(concat(arg, axis=0))
        else:
            cols.append(arg)

    return concat(cols, axis=1)


def vec(*args):
    """
    Create a one-dimensional array with the provided the scalars.

    Parameters
    ----------
    arg : Affine, Var, VarSub, Real, np.ndarray
        Each arg represents a scalar to be included into the one-dimensional array.

    Returns
    -------
    out : Affine
        A one-dimensional array created with the provided scalars.

    Notes
    -----
    The input arguments can be real numbers, NumPy arrays, or RSOME objects. All
    arguments must be scalars, i.e. their sizes must be one, otherwise the function
    raises an error message.
    """

    iters = []
    for arg in args:
        if isinstance(arg, Real):
            arg = np.array([arg])
        if isinstance(arg, np.ndarray):
            if arg.size != 1:
                raise ValueError('All inputs must have their sizes to be one.')
            arg = arg.reshape((1, ))
        else:
            arg = arg.to_affine()
            if arg.size != 1:
                raise ValueError('All inputs must have their sizes to be one.')
            arg = arg.reshape((1, ))

        iters.append(arg)

    return concat(iters)


class Model:
    """
    The Model class creates an LP model object.
    """

    def __init__(self, nobj=False, mtype='R', name=None, top=None):

        self.mtype = mtype
        self.top = top
        self.nobj = nobj
        self.name = name

        self.vars = []
        self.auxs = []
        self.last = 0
        self.constr_idx = 0
        self.ciarray = None
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
                constr.index = self.constr_idx
                self.constr_idx += 1
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

        return constr

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
            If primal=False, the method returns the dual formula.

        refresh : bool
            Leave the argument unspecified.

        obj : bool
            Leave the argument unspecified.

        Returns
        -------
        prog : rsome.lp.LinProg
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
                obj = obj.reshape(obj.size)
            else:
                # obj = np.ones((1, self.last))
                obj = np.ones(self.last)

            data_list = []
            indices_list = []
            indptr = [0]
            last = 0

            data_list += [item.linear.data
                          for item in self.lin_constr + self.aux_constr]
            indices_list += [item.linear.indices
                             for item in self.lin_constr + self.aux_constr]
            constr_idx_list = [np.array([item.index] * item.linear.shape[0])
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
                                  for item in self.lin_constr + self.aux_constr]

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

            ub = np.array([np.inf] * self.last)
            lb = np.array([-np.inf] * self.last)

            for b in self.bounds + self.aux_bounds:
                if b.btype == 'U':
                    ub[b.indices] = np.minimum(b.values, ub[b.indices])
                elif b.btype == 'L':
                    lb[b.indices] = np.maximum(b.values, lb[b.indices])

            formula = LinProg(linear, const, sense,
                              vtype, ub, lb, obj)
            self.primal = formula
            self.pupdate = False

            if constr_idx_list:
                self.ciarray = np.concatenate(constr_idx_list)
            else:
                self.ciarray = []

            return formula

        else:
            if self.dual is not None and not self.dupdate:
                return self.dual

            primal = self.do_math(obj=obj)
            if 'B' in primal.vtype or 'I' in primal.vtype:
                string = '\nIntegers detected.'
                string += '\nDual of the continuous relaxation is returned'
                warnings.warn(string)

            primal_linear = primal.linear
            primal_const = primal.const
            primal_sense = primal.sense
            indices_ub = np.where((primal.ub != 0) &
                                  (primal.ub != np.inf))[0]
            indices_lb = np.where((primal.lb != 0) &
                                  (primal.lb != - np.inf))[0]
            indices_fixed = np.where(primal.lb == primal.ub)[0]

            nub = len(indices_ub)
            nlb = len(indices_lb)
            nfixed = len(indices_fixed)
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
            if nfixed > 0:
                matrix_fixed = csr_matrix((np.array([-1] * nfixed), indices_fixed,
                                           np.arange(nfixed + 1)), (nfixed, nv))
                primal_linear = sp.vstack((primal_linear, matrix_fixed))
                primal_const = np.concatenate((primal_const,
                                               primal.lb[indices_fixed]))
                primal_sense = np.concatenate((primal_sense, np.ones(nfixed)))

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
            # dual_lb = - np.ones(ndv) * np.inf
            dual_lb = - np.array([np.inf] * ndv)

            indices_eq = np.where(primal_sense == 1)[0]
            if len(indices_eq):
                dual_ub[indices_eq] = np.inf

            if len(indices_neg) > 0:
                dual_linear[indices_neg, :] = - dual_linear[indices_neg, :]
                dual_const[indices_neg] = - dual_const[indices_neg]

            formula = LinProg(dual_linear, dual_const, dual_sense,
                              np.array(['C']*ndv), dual_ub, dual_lb, dual_obj)
            self.dual = formula
            self.dupdate = False

            return formula

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
                Mosek, and COPT.
            params : dict
                A dictionary that specifies parameters of the selected solver.
                So far the argument only applies to Gurobi, CPLEX, and Mosek.
        """

        if solver is None:
            solution = def_sol(self.do_math(obj=True), display, log, params)
        else:
            solution = solver.solve(self.do_math(obj=True), display, log, params)

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
            raise RuntimeError('The model is unsolved.')

        solution = self.solution
        if np.isnan(solution.objval):
            msg = 'No solution available. '
            msg += f'{solution.solver} solution status: {solution.status}.'
            raise RuntimeError(msg)

        return self.sign * self.solution.objval

    def optimal(self):

        if self.solution is None:
            return False
        else:
            return not np.isnan(self.solution.objval)


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
        """
        Returns an array containing the same variables with a new shape.

        Parameters
        ----------
        shape : tuple
            The new shape of the returned array.

        Returns
        -------
        out : Affine
            An array of the specified shape containing the given variables.
        """

        return self.to_affine().reshape(shape)

    def flatten(self):
        """
        Returns a 1D array containing the same variables.

        Returns
        -------
        out : Affine
            A 1D array of the given variables.
        """

        return self.to_affine().flatten()

    def diag(self, k=0, fill=False):
        """
        Return the diagonal elements of a 2-D array.

        Refer to `rsome.math.diag` for full documentation.

        See Also
        --------
        rsome.math.diag : equivalent function
        """

        return self.to_affine().diag(k, fill)

    def tril(self, k=0):
        """
        Return the lower triangular elements of a 2-D array. The remaining
        elements are filled with zeros.

        Refer to `rsome.math.tril` for full documentation.

        See Also
        --------
        rsome.math.tril : equivalent function
        """

        return self.to_affine().tril(k)

    def triu(self, k=0):
        """
        Return the upper triangular elements of a 2-D array. The remaining
        elements are filled with zeros.

        Refer to `rsome.math.triu` for full documentation.

        See Also
        --------
        rsome.math.triu : equivalent function
        """

        return self.to_affine().triu(k)

    def abs(self):

        return self.to_affine().abs()

    def norm(self, degree, method=None):
        """
        Return the first, second, or infinity norm of a 1-D array.

        Refer to `rsome.math.norm` for full documentation.

        See Also
        --------
        rsome.math.norm : equivalent function
        """

        return self.to_affine().norm(degree, method)

    def pnorm(self, degree, method=None):
        """
        Return the p-norm of a 1-D array, where p is a real number
        larger than 1.

        Refer to `rsome.math.pnorm` for full documentation

        See Also
        --------
        rsome.math.pnorm : equivalent function
        """

        return self.to_affine().pnorm(degree, method)

    def gmean(self, beta=None):
        """
        Return the weighted geometric mean of a 1-D array. The weights
        are specified by an array-like structure beta. It is expressed
        as prod(affine ** beta) ** (1/sum(beta))

        Refer to `rsome.gmean` for full documentation.

        See Also
        --------
        rso.gmean : equivalent function
        """

        return self.to_affine().gmean(beta)

    def square(self):
        """
        Return the element-wise square of an array.

        Refer to `rsome.math.square` for full documentation

        See Also
        --------
        rsome.lp.square : equivalent function
        """

        return self.to_affine().square()

    def power(self, p, q=1):
        """
        Return the element-wise integer power of the given affine
        array, i.e. affine ** (p/q)

        Refer to `rsome.power` for full documentation.

        See Also
        --------
        rso.power : equivalent function
        """

        return self.to_affine().power(p, q)

    def sumsqr(self):
        """
        Return the sum of squares of a 1-D array.

        Refer to `rsome.math.sumsqr` for full documentation.

        See Also
        --------
        rsome.math.sumsqr : equivalent function
        """

        return self.to_affine().sumsqr()

    def quad(self, qmat):
        """
        Return the quadratic expression var @ qmat @ var.

        Refer to `rsome.math.quad` for full documentation.

        See Also
        --------
        rsome.math.quad : equivalent function
        """

        return self.to_affine().quad(qmat)

    def rsocone(self, y, z):
        """
        Return the rotated second-order cone constraint.

        Refer to `rsome.math.rsocone` for full documentation.

        See Also
        --------
        rsome.math.rsocone : equivalent function
        """

        return self.to_affine().rsocone(y, z)

    def expcone(self, x, z):
        """
        Return the exponential cone constraint z*exp(x/z) <= var.

        Refer to `rsome.math.expcone` for full documentation.

        See Also
        --------
        rsome.math.expcone : equivalent function
        """

        return self.to_affine().expcone(x, z)

    def exp(self):
        """
        Return the natural exponential function exp(var).

        Refer to `rsome.math.exp` for full documentation.

        See Also
        --------
        rsome.math.exp : equivalent function
        """

        return self.to_affine().exp()

    def pexp(self, scale):
        """
        Return the perspective natural exponential function
        scale * exp(var/scale).

        Refer to `rsome.math.pexp` for full documentation.

        See Also
        --------
        rsome.math.pexp : equivalent function
        """

        return self.to_affine().pexp(scale)

    def log(self):
        """
        Return the natural logarithm function log(var).

        Refer to `rsome.math.log` for full documentation.

        See Also
        --------
        rsome.math.log : equivalent function
        """

        return self.to_affine().log()

    def plog(self, scale):
        """
        Return the perspective of natural logarithm function
        scale * log(var/scale).

        Refer to `rsome.math.plog` for full documentation.

        See Also
        --------
        rsome.math.plog : equivalent function
        """

        return self.to_affine().plog(scale)

    def entropy(self):
        """
        Return the natural exponential function -sum(var*log(var)).

        Refer to `rsome.math.entropy` for full documentation.

        See Also
        --------
        rsome.math.entropy : equivalent function
        """

        return self.to_affine().entropy()

    def softplus(self):
        """
        Return the softplus function log(1 + exp(var)).

        Refer to `rsome.math.softplus` for full documentation.

        See Also
        --------
        rsome.math.softplus : equivalent function
        """

        return self.to_affine().softplus()

    def kldiv(self, q, r):
        """
        Return the KL divergence constraints sum(var*log(var/q)) <= r.

        Refer to `rsome.math.kldiv` for full documentation.

        See Also
        --------
        rsome.math.kldiv : equivalent function
        """

        return self.to_affine().kldiv(q, r)

    def trace(self):
        """
        Return the trace of a 2D array.

        Refer to `rsome.lp.trace` for full documentation.

        See Also
        --------
        rsome.lp.rsocone : equivalent function
        """

        return self.to_affine().trace()

    def logdet(self):
        """
        Return the log-determinant of a positive semidefinite matrix
        expressed as a two-dimensional array.

        Refer to `rsome.logdet` for full documentation.

        See Also
        --------
        rsome.logdet : equivalent function
        """

        return self.to_affine().logdet()

    def rootdet(self):
        """
        Return the root-determinant of a positive semidefinite matrix
        expressed as a two-dimensional array. The root-determinant is
        expressed as (det(A))**(1/L), where L is the dimension of the
        two-dimensinoal array.

        Refer to `rsome.rootdet` for full documentation.

        See Also
        --------
        rsome.rootdet : equivalent function
        """

        return self.to_affine().rootdet()

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
            raise RuntimeError('The model is unsolved.')

        solution = self.model.solution
        if np.isnan(solution.objval):
            msg = 'No solution available. '
            msg += f'{solution.solver} solution status: {solution.status}.'
            raise RuntimeError(msg)

        indices = range(self.first, self.first + self.size)
        var_sol = np.array(solution.x)[indices]
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

        cond1 = isinstance(other, (Real, np.ndarray)) or sp.issparse(other)
        cond2 = self.model.mtype not in 'EP'
        if cond1 and cond2:
            upper = other + np.zeros(self.shape)
            upper = upper.reshape((upper.size, ))
            indices = np.arange(self.first, self.first + self.size,
                                dtype=np.int32)
            return Bounds(self.model, indices, upper, 'U')
        else:
            return self.to_affine() <= other

    def __ge__(self, other):

        cond1 = isinstance(other, (Real, np.ndarray)) or sp.issparse(other)
        cond2 = self.model.mtype not in 'EP'
        if cond1 and cond2:
            lower = other + np.zeros(self.shape)
            lower = lower.reshape((lower.size, ))
            indices = np.arange(self.first, self.first + self.size,
                                dtype=np.int32)
            return Bounds(self.model, indices, lower, 'L')
        else:
            return self.to_affine() >= other

    def __eq__(self, other):

        return self.to_affine() == other

    def __rshift__(self, other):

        return self.to_affine().__rshift__(other)

    def __lshift__(self, other):

        return self.to_affine().__lshift__(other)

    def assign(self, values):

        if self.model.mtype != 'S':
            raise ValueError('Unsupported variables.')
        else:
            if not isinstance(values, (np.ndarray, Real)):
                raise TypeError('The second argument must be numerical values.')

            values = np.array(values, dtype=float) + np.zeros(self.shape, dtype=float)

            return RandVal(self, values.reshape(self.shape))

    def __call__(self):

        return self.to_affine()()


class VarSub(Vars):
    """
    The VarSub class creates a variable array with subscript indices.
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

        if isinstance(other, Real):
            upper = upper = super().__le__(other)
            indices = self.indices.reshape((self.indices.size, ))
            bound_indices = upper.indices.reshape((upper.indices.size, ))[indices]
            bound_values = upper.values.reshape(upper.values.size)[indices]
            return Bounds(upper.model, bound_indices, bound_values, 'U')
        else:
            return self.to_affine().__le__(other)

    def __ge__(self, other):

        if isinstance(other, Real):
            lower = super().__ge__(other)
            indices = self.indices.reshape((self.indices.size, ))
            bound_indices = lower.indices.reshape((lower.indices.size, ))[indices]
            bound_values = lower.values.reshape((lower.indices.size, ))[indices]
            return Bounds(lower.model, bound_indices, bound_values, 'L')
        else:
            return self.to_affine().__ge__(other)

    def __call__(self):

        return self.to_affine()()


class Affine:
    """
    The Affine class creates an array of affine expressions.
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
        size = self.size

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
        """
        Returns an array containing the same affine expressions with a
        new shape.

        Parameters
        ----------
        shape : tuple
            The new shape of the returned array.

        Returns
        -------
        out : Affine
            An array of the specified shape containing the given affine
            expressions.
        """

        if isinstance(self.const, np.ndarray):
            new_const = self.const.reshape(shape)
        else:
            new_const = np.array([self.const]).reshape(shape)
        return Affine(self.model, self.linear, new_const)

    def flatten(self):
        """
        Returns a 1D array containing the same affine expressions.

        Returns
        -------
        out : Affine
            A 1D array of the given affine expressions.
        """

        return self.reshape((self.size, ))

    def diag(self, k=0, fill=False):
        """
        Return the diagonal elements of a 2-D array.

        Refer to `rsome.math.diag` for full documentation.

        See Also
        --------
        rsome.math.diag : equivalent function
        """

        if len(self.shape) != 2:
            raise ValueError('The diag function can only be applied to 2D arrays.')

        num = min(self.shape)
        if k >= 0:
            idx_row = np.arange(num - k)
            idx_col = np.arange(k, num)
        else:
            idx_row = np.arange(-k, num)
            idx_col = np.arange(num + k)

        if fill:
            bool_mat = np.ones(self.shape, dtype=bool)
            bool_mat[idx_row, idx_col] = False
            bool_idx = bool_mat.flatten()

            affine = self + 0
            affine.linear = lil_matrix(affine.linear)
            affine.linear[bool_idx] = 0.0
            affine.linear = csr_matrix(affine.linear)
            affine.const = np.diag(np.diag(affine.const, k), k)

            return affine
        else:
            return self[idx_row, idx_col]

    def tril(self, k=0):
        """
        Return the lower triangular elements of a 2-D array. The remaining
        elements are filled with zeros.

        Refer to `rsome.math.tril` for full documentation.

        See Also
        --------
        rsome.math.tril : equivalent function
        """

        if len(self.shape) != 2:
            raise ValueError('The tril function can only be applied to 2D arrays.')

        bool_idx = (~np.tril(np.ones(self.shape, dtype=bool), k)).flatten()

        affine = self + 0
        affine.linear = lil_matrix(affine.linear)
        affine.linear[bool_idx] = 0.0
        affine.linear = csr_matrix(affine.linear)
        affine.const = np.tril(affine.const, k)

        return affine

    def triu(self, k=0):
        """
        Return the upper triangular elements of a 2-D array. The remaining
        elements are filled with zeros.

        Refer to `rsome.math.triu` for full documentation.

        See Also
        --------
        rsome.math.triu : equivalent function
        """

        if len(self.shape) != 2:
            raise ValueError('The tril function can only be applied to 2D arrays.')

        bool_idx = (~np.triu(np.ones(self.shape, dtype=bool), k)).flatten()

        affine = self + 0
        affine.linear = lil_matrix(affine.linear)
        affine.linear[bool_idx] = 0.0
        affine.linear = csr_matrix(affine.linear)
        affine.const = np.triu(affine.const, k)

        return affine

    def sum(self, axis=None):

        if self.sparray is None:
            self.sparray = self.sv_array()

        indices = self.sparray.sum(axis=axis)

        linear = sv_to_csr(indices) @ self.linear
        const = self.const.sum(axis=axis)

        return Affine(self.model, linear, const)

    def trace(self):
        """
        Return the trace of a 2D array.

        Refer to `rsome.math.trace` for full documentation.

        See Also
        --------
        rsome.math.rsocone : equivalent function
        """

        if len(self.shape) != 2:
            raise ValueError('The trace function only applies to two-dimensional arrays')
        dim = min(self.shape)

        out = self[range(dim), range(dim)].sum()

        return out

    def __abs__(self):

        return Convex(self, np.zeros(self.shape), 'A', 1)

    def abs(self):

        return self.__abs__()

    def norm(self, degree, method=None):
        """
        Return the first, second, or infinity norm of a 1-D array.

        Refer to `rsome.math.norm` for full documentation

        See Also
        --------
        rsome.math.norm : equivalent function
        """

        if len(self.shape) != 1:
            err = 'Improper number of dimensions to norm. '
            err += 'The array must be 1-D.'
            raise ValueError(err)

        new_shape = ()
        if degree == 1:
            return Convex(self, np.zeros(new_shape), 'M', 1)
        elif degree == np.inf or degree == 'inf':
            return Convex(self, np.zeros(new_shape), 'I', 1)
        elif degree == 2:
            return Convex(self, np.zeros(new_shape), 'E', 1)
        else:
            # raise ValueError('Invalid norm order for the array.')
            return self.pnorm(degree, method)

    def pnorm(self, degree, method=None):
        """
        Return the p-norm of a 1-D array, where p is a real number
        larger than 1.

        Refer to `rsome.math.pnorm` for full documentation

        See Also
        --------
        rsome.math.pnorm : equivalent function
        """

        if len(self.shape) != 1:
            err = 'Improper number of dimensions to norm. '
            err += 'The array must be 1-D.'
            raise ValueError(err)

        new_shape = ()
        if method is None:
            if isinstance(degree, (int, Iterable)):
                method = 'soc'
            elif isinstance(degree, float):
                method = 'exc'
            else:
                raise TypeError('The degree parameter must be a real number.')

        if isinstance(degree, Iterable):
            a, b = degree
            if a <= b:
                raise ValueError('The degree parameter a/b must be larger than one.')
            if not (isinstance(a, int) and isinstance(b, int)):
                raise TypeError('The coefficients a and b must be integers.')
        elif isinstance(degree, Real):
            if degree <= 1:
                raise ValueError('The degree parameter must be larger than 1.')
        else:
            raise TypeError('The degree parameter can only one real number or two integers.')

        if method == 'soc':
            if not isinstance(degree, (int, Iterable)):
                raise TypeError('Unsupported degree for second-order conic expressions.')
            return Convex(self, np.zeros(new_shape), 'G', 1, params=degree)
        elif method == 'exc':
            if isinstance(degree, Iterable):
                a, b = degree
                degree = a / b
            return Convex(self, np.zeros(new_shape), 'N', 1, params=degree)
        else:
            raise ValueError("The method can only be 'soc' or 'exc'.")

    def gmean(self, beta=None):
        """
        Return the weighted geometric mean of a 1-D array. The weights
        are specified by an array-like structure beta. It is expressed
        as prod(affine ** beta) ** (1/sum(beta))

        Refer to `rsome.gmean` for full documentation.

        See Also
        --------
        rso.gmean : equivalent function
        """

        if len(self.shape) != 1:
            err = 'Improper number of dimensions for geometric mean. '
            err += 'The array must be 1-D.'
            raise ValueError(err)
        if beta is None:
            beta = [1] * self.size
        beta_array = np.array(beta)
        if (beta_array % beta_array.astype(int) > 0).any():
            raise ValueError('All beta values must be integers.')
        if len(beta_array.shape) != 1:
            err = 'Improper number of dimensions for beta values. '
            err += 'It must be 1-D.'
            raise ValueError(err)
        if (beta_array < 1).any():
            raise ValueError('All beta values must be no smaller than one.')
        if beta_array.size != self.size:
            raise ValueError('The sizes of the array and beta values do not match.')

        return Convex(self, np.zeros(1), 'C', -1, params=beta)

    def square(self):
        """
        Return the element-wise square of an array.

        Refer to `rsome.math.square` for full documentation

        See Also
        --------
        rsome.math.square : equivalent function
        """

        size = self.size
        shape = self.shape

        return Convex(self.reshape((size,)), np.zeros(shape), 'S', 1)

    def power(self, p, q=1):
        """
        Return the element-wise integer power of the given affine
        array, i.e. affine ** (p/q)

        Refer to `rsome.power` for full documentation.

        See Also
        --------
        rso.power : equivalent function
        """

        p_array, q_array = np.array(p), np.array(q)

        if (p_array == q_array).all():
            return self.__abs__()
        elif (p_array < q_array).any():
            raise ValueError('Exponent values must be no smaller than one.')

        if (p_array % p_array.astype(int) > 0).any():
            raise TypeError('RSOME only supports integer exponents.')
        if (q_array % q_array.astype(int) > 0).any():
            raise TypeError('RSOME only supports integer exponents.')

        shape = np.broadcast(np.zeros(self.shape), p_array, q_array).shape

        return Convex(self, np.zeros(shape), 'T', 1, params=(p_array, q_array))

    def sumsqr(self):
        """
        Return the sum of squares of a 1-D array.

        Refer to `rsome.math.sumsqr` for full documentation.

        See Also
        --------
        rsome.math.sumsqr : equivalent function
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
        Return the quadratic expression var @ qmat @ var.

        Refer to `rsome.math.quad` for full documentation.

        See Also
        --------
        rsome.math.quad : equivalent function
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

    def rsocone(self, y, z):
        """
        Return the rotated second-order cone constraint.

        Refer to `rsome.math.rsocone` for full documentation.

        See Also
        --------
        rsome.math.rsocone : equivalent function
        """

        if self.size > 1:
            if self.size != max(self.shape):
                err = 'Improper number of dimensions to norm. '
                err += 'The array must be a vector.'
                raise ValueError(err)

        if isinstance(y, (Vars, VarSub, Affine)):
            y = y.to_affine()
            if y.size > 1:
                raise ValueError('The expression of x must be a scalar.')
        else:
            raise TypeError('Unsupoorted type for rotated cone. ')
        if isinstance(y, (Vars, VarSub, Affine)):
            if self.model is not y.model:
                raise ValueError('Models mismatch.')

        if isinstance(z, (Vars, VarSub, Affine)):
            z = z.to_affine()
            if z.size > 1:
                raise ValueError('The expression of z must be a scalar.')
        else:
            raise TypeError('Unsupoorted type for rotated cone. ')
        if isinstance(z, (Vars, VarSub, Affine)):
            if self.model is not z.model:
                raise ValueError('Models mismatch.')

        affine_in = concat((((y-z)*0.5).reshape((1,)), self))
        affine_out = - ((y+z)*0.5).reshape((1,))

        return CvxConstr(self.model, affine_in, affine_out, multiplier=1, xtype='E')

    def expcone(self, x, z):
        """
        Return the exponential cone constraint z*exp(x/z) <= affine.

        Refer to `rsome.math.expcone` for full documentation.

        See Also
        --------
        rsome.math.expcone : equivalent function
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
        Return the natural exponential function exp(affine).

        Refer to `rsome.math.exp` for full documentation.

        See Also
        --------
        rsome.math.exp : equivalent function
        """

        return Convex(self, np.zeros(self.shape), 'X', 1)

    def pexp(self, scale):
        """
        Return the perspective natural exponential function
        scale * exp(affine/scale).

        Refer to `rsome.math.pexp` for full documentation.

        See Also
        --------
        rsome.math.pexp : equivalent function
        """

        return PerspConvex(self, scale, np.zeros(self.shape), 'X', 1)

    def log(self):
        """
        Return the natural logarithm function log(affine).

        Refer to `rsome.math.log` for full documentation.

        See Also
        --------
        rsome.math.log : equivalent function
        """

        return Convex(self, np.zeros(self.shape), 'L', -1)

    def plog(self, scale):
        """
        Return the perspective of natural logarithm function
        scale * log(affine/scale).

        Refer to `rsome.math.plog` for full documentation.

        See Also
        --------
        rsome.math.plog : equivalent function
        """

        return PerspConvex(self, scale, np.zeros(self.shape), 'L', -1)

    def entropy(self):
        """
        Return the natural exponential function -sum(affine*log(affine)).

        Refer to `rsome.math.entropy` for full documentation.

        See Also
        --------
        rsome.math.entropy : equivalent function
        """

        if self.shape != ():
            if self.size != max(self.shape):
                raise ValueError('The expression must be a vector.')

        return Convex(self, np.float64(0), 'P', -1)

    def softplus(self):
        """
        Return the softplus function log(1 + exp(var)).

        Refer to `rsome.math.softplus` for full documentation.

        See Also
        --------
        rsome.math.softplus : equivalent function
        """

        return Convex(self, np.zeros(self.shape), 'F', 1)

    def kldiv(self, q, r):
        """
        Return the KL divergence constraints sum(affine*log(affine/q)) <= r.

        Refer to `rsome.math.kldiv` for full documentation.

        See Also
        --------
        rsome.math.kldiv : equivalent function
        """

        affine = self.to_affine().reshape((self.size, ))

        if isinstance(q, Real):
            q = np.array([q]*self.size)
        elif isinstance(q, np.ndarray):
            if q.size == 1:
                q = np.array([q.flatten()[0]] * self.size)
            else:
                q = q.reshape(affine.shape)
        elif isinstance(q, (Vars, VarSub, Affine)):
            if affine.model is not q.model:
                raise ValueError('Models mismatch.')
            if q.size == 1:
                q = q * np.ones(affine.shape)
            else:
                q = q.reshape(affine.shape)

        return KLConstr(affine, q, r)

    def logdet(self):
        """
        Return the log-determinant of a positive semidefinite matrix
        expressed as a two-dimensional array.

        Refer to `rsome.logdet` for full documentation.

        See Also
        --------
        rsome.logdet : equivalent function
        """

        new_shape = ()

        return Convex(self, np.zeros(new_shape), 'O', -1)

    def rootdet(self):
        """
        Return the root-determinant of a positive semidefinite matrix
        expressed as a two-dimensional array. The root-determinant is
        expressed as (det(A))**(1/L), where L is the dimension of the
        two-dimensinoal array.

        Refer to `rsome.rootdet` for full documentation.

        See Also
        --------
        rsome.rootdet : equivalent function
        """

        new_shape = ()

        return Convex(self, np.zeros(new_shape), 'D', -1)

    def concat(self, other, axis=0):

        if not isinstance(other, Affine):
            raise TypeError('Incorrect type in concatenation.')
        if self.model != other.model:
            raise ValueError('Model mismatch.')

        idx_left = np.arange(self.size).reshape(self.shape)
        idx_other = np.arange(self.size, self.size + other.size).reshape(other.shape)
        idx_all = np.concatenate((idx_left, idx_other), axis=axis).flatten()

        if self.linear.shape[1] < other.linear.shape[1]:
            self.linear.resize(self.linear.shape[0], other.linear.shape[1])
        if self.linear.shape[1] > other.linear.shape[1]:
            other.linear.resize(other.linear.shape[0], self.linear.shape[1])
        linear_all = sp.vstack((self.linear, other.linear))[idx_all]
        const_all = np.concatenate((self.const, other.const), axis=axis)

        return Affine(self.model, linear_all, const_all)

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

        return self.__add__(-other)

    def __rsub__(self, other):

        return (-self).__add__(other)

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

    def __rshift__(self, other):

        left = self - other
        if len((left.shape)) != 2:
            msg = """Only two-dimensional arrays can be used to construct linear
            matrix equality constraints."""
            raise ValueError(msg)
        if left.shape[0] != left.shape[1]:
            msg = """Only two-dimensional square arrays can be used to construct
            linear matrix equality constraints."""
            raise ValueError(msg)

        return LMIConstr(left.model, left.linear, -left.const, left.shape[0])

    def __lshift__(self, other):

        return (-self).__rshift__(-other)

    def __call__(self):

        if self.model.mtype != 'R':
            raise ValueError('Unsupported affine expression.')

        if self.model.solution is None:
            raise SyntaxError('No available solution!')
        else:
            linear = self.linear
            const = self.const
            nvar = linear.shape[1]

            x = self.model.solution.x[:nvar]
            output = (linear @ x).reshape(self.shape)
            output += const.reshape(self.shape)

            if output.ndim == 0:
                output = output.item()

            return output


class Convex:
    """
    The Convex class creates an object of convex functions.
    """

    __array_priority__ = 101

    def __init__(self, affine_in, affine_out, xtype, sign,
                 multiplier=1, sum_axis=False, params=None):

        self.model = affine_in.model
        self.affine_in = affine_in
        self.affine_out = affine_out
        self.multiplier = multiplier
        self.sum_axis = sum_axis
        self.params = params
        self.size = affine_out.size
        self.xtype = xtype
        self.sign = sign

    def __repr__(self):
        xtypes = {'A': 'absolute expression',
                  'M': 'one-norm expression',
                  'N': 'general Lp-norm expression',       # Expressed by exponential cones
                  'G': 'general Lp-norm expression',       # Expressed by second-order cones
                  'D': 'root determinant',                 # Expressed by second-order cones
                  'O': 'log determinant',                  # Expressed by exponential cones
                  'T': 'power expression',                 # Expressed by second-order cones
                  'C': 'geometric mean',                   # Expressed by second-order cones
                  'E': 'Eclidean norm expression',
                  'I': 'infinity norm expression',
                  'S': 'element-wise square expression',
                  'Q': 'sum of squares expression',
                  'X': 'natural exponential expression',
                  'L': 'natural logarithm expression',
                  'F': 'softplus function',
                  'P': 'entropy expression',
                  'K': 'KL divergence expression',
                  'W': 'piecewise linear expression'}
        if self.affine_out.shape == ():
            shapes = 'an' if self.xtype in 'AEISP' else 'a'
        else:
            shapes = 'x'.join([str(dim) for dim in self.affine_out.shape])

        suffix = 's' if self.size > 1 else ''
        string = shapes + ' ' + xtypes[self.xtype] + suffix

        return string

    def __neg__(self):

        return Convex(self.affine_in, -self.affine_out, self.xtype, -self.sign,
                      self.multiplier,
                      params=self.params)

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
                            self.xtype, self.sign, self.multiplier,
                            params=self.params)

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

        if self.xtype in 'AMNGIEXLPFKODTC':
            multiplier = self.multiplier * abs(other)
        elif self.xtype in 'SQ':
            multiplier = self.multiplier * abs(other) ** 0.5
        else:
            raise ValueError('Unknown type of convex function.')

        return Convex(self.affine_in, other * self.affine_out,
                      self.xtype, np.sign(other)*self.sign, multiplier,
                      params=self.params)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __le__(self, other):

        left = self - other
        if left.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return CvxConstr(left.model, left.affine_in, left.affine_out,
                         left.multiplier, left.xtype, params=left.params)

    def __ge__(self, other):

        right = other - self
        if right.sign == -1:
            raise ValueError('Nonconvex constraints.')

        return CvxConstr(right.model, right.affine_in, right.affine_out,
                         right.multiplier, right.xtype, params=right.params)

    def __eq__(self, other):

        raise TypeError('Convex expressions are not applied to equality constraints')

    def sum(self, axis=None):

        if self.xtype not in 'XL':
            raise ValueError('Convex functions do not support the sum() method.')

        return Convex(self.affine_in, self.affine_out.sum(axis=axis),
                      self.xtype, self.sign, self.multiplier, axis, params=self.params)

    def __call__(self):

        if self.model.mtype != 'R':
            raise ValueError('Unsupported affine expression.')

        if self.model.solution is None:
            raise SyntaxError('No available solution!')
        else:
            value_in = self.affine_in()
            if isinstance(self.affine_out, Affine):
                value_out = self.affine_out()
            else:
                value_out = self.affine_out

            if self.xtype == 'A':
                output = self.multiplier*self.sign*abs(value_in) + value_out
            elif self.xtype == 'M':
                output = self.multiplier*self.sign*abs(value_in).sum() + value_out
            elif self.xtype in 'NG':
                d = self.params
                if isinstance(d, Iterable):
                    d = d[0] / d[1]
                output = self.multiplier*self.sign*np.linalg.norm(abs(value_in), d) + value_out
            elif self.xtype == 'E':
                output = self.multiplier*self.sign*((value_in**2).sum())**0.5 + value_out
            elif self.xtype == 'I':
                output = self.multiplier*self.sign*abs(value_in).max() + value_out
            elif self.xtype == 'S':
                output = self.multiplier**2*self.sign*(value_in**2) + value_out
            elif self.xtype == 'Q':
                output = self.multiplier**2*self.sign*(value_in**2).sum() + value_out
            elif self.xtype == 'X':
                output = self.multiplier*self.sign*np.exp(value_in) + value_out
            elif self.xtype == 'L':
                output = - self.multiplier*self.sign*np.log(value_in) + value_out
            elif self.xtype == 'F':
                output = self.multiplier*self.sign*np.log(1+np.exp(value_in)) + value_out
            elif self.xtype == 'P':
                output = self.multiplier*self.sign*(value_in * np.log(1/value_in)).sum()
                output += value_out
            elif self.xtype == 'T':
                expo = self.params[0] / self.params[1]
                output = self.multiplier*self.sign*(value_in ** expo) + value_out
                output += value_out
            else:
                raise ValueError('Unsupported convex/concave expression.')

            return output


class PiecewiseConvex:
    """
    The PiecewiseConvex class creates an object of piecewise functions.
    """

    def __init__(self, model, pieces, sign=1, add_sign=1):

        self.model = model
        self.pieces = pieces
        self.sign = sign
        self.add_sign = add_sign

    def __repr__(self):

        num_pieces = len(self.pieces)
        cvx = 'convex' if self.sign > 0 else 'concave'

        return f'a {cvx} piecewise function with {num_pieces} pieces'

    def __neg__(self):

        return PiecewiseConvex(self.model, self.pieces, -self.sign, self.add_sign)

    def __add__(self, other):

        if isinstance(other, (np.ndarray, Vars, VarSub, Affine, RoAffine)):
            if other.size != 1:
                raise ValueError('Expressions in piecewise functions must be scalars.')
        elif isinstance(other, (DecRule, DecRuleSub)):
            other = other.to_affine()
            if other.size != 1:
                raise ValueError('Expressions in piecewise functions must be scalars.')
        elif not isinstance(other, Real):
            raise TypeError('Unsupported expressions.')

        pieces = [piece + other*self.sign for piece in self.pieces]

        return PiecewiseConvex(self.model, pieces, self.sign, self.add_sign)

    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):

        return self.__add__(-other)

    def __rsub__(self, other):

        return (-self).__add__(other)

    def __mul__(self, other):

        if not isinstance(other, Real):
            raise TypeError('Incorrect syntax.')

        other_sign = np.sign(other)
        other_abs = abs(other)

        pieces = [piece*other_abs for piece in self.pieces]

        return PiecewiseConvex(self.model, pieces, self.sign*other_sign, self.add_sign)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __le__(self, other):

        left = self - other
        if left.sign == -1:
            raise ValueError('Nonconvex constraints.')

        pieces = [piece <= 0 for piece in left.pieces]

        return PWConstr(left.model, pieces)

    def __ge__(self, other):

        right = other - self
        if right.sign == -1:
            raise ValueError('Nonconvex constraints.')

        pieces = [piece <= 0 for piece in right.pieces]

        return PWConstr(right.model, pieces)

    def __getitem__(self, item):

        return self.pieces[item]

    def __len__(self):

        return len(self.pieces)

    @property
    def E(self):

        return ExpPiecewiseConvex(self.model, self.pieces, self.sign, self.add_sign)


class PerspConvex(Convex):
    """
    The PerspConvex object creates an object of a perspective convex function.
    """

    def __init__(self, affine_in, affine_scale, affine_out, xtype, sign,
                 multiplier=1):

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
        xtype = xtypes[self.xtype]
        string = f"{shapes} perspective expression{suffix} of the {xtype}"
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
    The Roaffine class creats an object of uncertain affine functions.
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
                sparray = (np.arange(self.size).reshape(self.shape) +
                           np.zeros(other.shape))
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

    def __call__(self, *args):

        nrand = self.rand_model.last
        rvec = np.zeros(nrand)
        for arg in args:
            if not isinstance(arg, RandVal):
                raise TypeError('Unsupported type for defining random variable values.')

            index = range(arg.rvar.first, arg.rvar.last)
            rvec[index] = arg.values.ravel()

        raffine_value = self.raffine()
        affine_value = self.affine()

        nrand = raffine_value.shape[1]
        output = (raffine_value@rvec[:nrand]).reshape(self.shape) + affine_value

        return output


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
        self.index = None

    def __repr__(self):

        size = self.linear.shape[0]
        if size == 1:
            return '1 linear constraint'
        else:
            return '{} linear constraints'.format(size)

    def dual(self):

        if self.model.solution is None:
            raise RuntimeError('The model is unsolved. ')

        cidx = self.index
        if cidx is None:
            raise RuntimeError('The constraint is not a part of any model. ')

        solution = self.model.solution
        if solution.y is None:
            msg = 'The dual solution is not available. '
            msg += f'{solution.solver} solution status: {solution.status}.'
            warnings.warn(msg)
        else:
            dual_sol = solution.y['pi'][self.model.ciarray == cidx] * self.model.sign
            if dual_sol.size == 1:
                dual_sol = dual_sol.item()

            return dual_sol


class LMIConstr:
    """
    The LMIConstr class creates an object of linear matrix inequality
    constraints.
    """

    def __init__(self, model, linear, const, dim):

        self.model = model
        self.linear = linear
        self.const = const
        self.dim = dim

    def __repr__(self):

        dim = int(self.linear.shape[0] ** 0.5)
        return f'{dim}x{dim} linear matrix inequliaty constraint'


class CvxConstr:
    """
    The CvxConstr class creates an object of convex constraints.
    """

    def __init__(self, model, affine_in, affine_out, multiplier, xtype, params=None):

        self.model = model
        self.affine_in = affine_in
        self.affine_out = affine_out
        self.multiplier = multiplier
        self.xtype = xtype
        self.params = params

    def __repr__(self):

        size = self.affine_out.size
        if size == 1:
            return '1 convex constraint'
        else:
            return '{} convex constraints'.format(size)


class PWConstr:
    """
    The PWConstr class creates an object of piecewise convex constraints.
    """

    def __init__(self, model, pieces, supp_set=None):

        self.model = model
        self.pieces = pieces
        self.supp_set = supp_set

    def forall(self, *args):

        pieces = []
        for piece in self.pieces:
            if isinstance(piece, (DecLinConstr, DecBounds, RoConstr)):
                pieces.append(piece.forall(*args))
            else:
                pieces.append(piece)

        return PWConstr(self.model, pieces, args)


class PCvxConstr(CvxConstr):

    def __init__(self, model, affine_in, affine_scale, affine_out,
                 multiplier, xtype):

        super().__init__(model, affine_in, affine_out, multiplier, xtype)
        self.affine_scale = affine_scale


class Bounds:
    """
    The Bounds class creates an object for upper or lower bounds.
    """

    def __init__(self, model, indices, values, btype):

        self.model = model
        self.indices = indices
        self.values = values
        self.btype = btype

    def dual(self):

        if self.model.solution is None:
            raise RuntimeError('The model is unsolved. ')

        solution = self.model.solution
        if solution.y is None:
            msg = 'The dual solution is not available. '
            msg += f'{solution.solver} solution status: {solution.status}.'
            warnings.warn(msg)
        else:
            primal = self.model.primal
            if self.btype == 'U':
                pi = self.model.solution.y['upi'] * self.model.sign
                output = pi[self.indices]
                output[primal.ub[self.indices] < self.values] = 0
            elif self.btype == 'L':
                pi = self.model.solution.y['lpi'] * self.model.sign
                output = pi[self.indices]
                output[primal.lb[self.indices] > self.values] = 0
            else:
                raise ValueError('Unknown bounds. ')

            if output.size == 1:
                output = output.item()

            return output


class ConeConstr:
    """
    The ConeConstr class creates an object of second-order cone constraints.
    """

    def __init__(self, model, left_var, left_index, right_var, right_index):

        self.model = model
        self.left_var = left_var
        self.right_var = right_var
        self.left_index = left_index
        self.right_index = right_index


class ExpConstr:
    """
    The ExpConstr class creates an object of exponential cone constraints.
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
    The KLConstr class creates an object of constraint for KL divergence.
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


class IPCone:

    def __init__(self, x, r, beta):

        if x.model != r.model:
            raise ValueError('Model mismatch.')
        self.model = x.model
        self.left = x.to_affine()
        if self.left != 1:
            raise ValueError('Variable dimension')
        self.right = r.flatten()
        if self.right.size != len(beta):
            raise ValueError('Variable dimension mismatches degrees.')

        self.beta = list(beta)

        self.branches = None

    def __repr__(self):

        return f"betas: {self.beta}"

    def __str__(self):

        return self.__repr__()

    def to_pot(self):

        model = self.model

        beta = self.beta.copy()
        degree = sum(beta)

        xbeta = int(2 ** np.ceil(np.log2(degree)) - degree)

        if xbeta > 0:
            s = model.dvar(aux=True).flatten()
            right = concat((self.right, s))
            beta.append(xbeta)
            return IPCone(s, right, beta), [s >= abs(self.left)]
        else:
            return IPCone(self.left, self.right, beta), []

    def split(self):

        model = self.model

        beta = self.beta
        degree = sum(beta)
        left = self.left
        right = self.right

        if len(beta) == 2 and beta[0] == beta[1]:
            return [left.rsocone(right[0], right[1])]
        elif max(beta) >= degree/2:
            index = np.argmax(beta)
            mid = beta[index] - degree//2
            beta1 = beta[:index] + ([] if mid == 0 else [mid]) + beta[index+1:]
            if mid > 0:
                right1 = right
            else:
                idx = list(range(len(beta)))
                idx.remove(index)
                right1 = right[idx]

            u = model.dvar()

            b1 = IPCone(u, right1, beta1)

            self.branches = [b1]

            constr = [left.rsocone(u, right[index])]
            constr.extend(b1.split())

            return constr

        else:
            cum = np.cumsum(beta)
            index = np.argmax(cum >= degree/2)

            mid = degree//2 - cum[index-1]
            beta1 = beta[:index] + [mid]
            right1 = right[:index+1]
            if mid == beta[index]:
                beta2 = beta[index+1:]
                right2 = right[index+1:]
            else:
                beta2 = [beta[index] - mid] + beta[index+1:]
                right2 = right[index:]

            u = model.dvar(aux=True)
            v = model.dvar(aux=True)

            b1 = IPCone(u, right1, beta1)
            b2 = IPCone(v, right2, beta2)

            self.branches = [b1, b2]

            constr = [left.rsocone(u, v)]
            constr.extend(b1.split() + b2.split())
            return constr

    def to_soc(self):

        if len(self.beta) == 1:
            return [self.left.abs() <= self.right]
        else:
            ip_cone, constr = self.to_pot()
            constr.extend(ip_cone.split())
            return constr


class RoConstr:
    """
    The Roaffine class creats an object of uncertain affine functions.
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

        bounds = []
        index_pos = (support.ub == 0)
        if any(index_pos):
            bounds.append(dual_var[:, index_pos] <= 0)
        index_neg = (support.lb == 0)
        if any(index_neg):
            bounds.append(dual_var[:, index_neg] >= 0)

        if num_rand == support.linear.shape[0]:
            constr_list = [constr1, constr2]
            constr_list += [] if bounds is None else bounds
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
            for pconstr in support.lmi:
                dim = pconstr['dim']
                symat = (pconstr['linear']@dual_var[n]).reshape((dim, dim))
                symat -= pconstr['const']
                constr_list.append(symat >> 0)

        return constr_list


class DecVar(Vars):
    """
    The DecVar class creates an object of generic variable array
    (here-and-now or wait-and-see) for adaptive DRO models.
    """

    def __init__(self, dro_model, dvars, fixed=True, name=None):

        super().__init__(dvars.model, dvars.first, dvars.shape,
                         dvars.vtype, dvars.name)
        self.dro_model = dro_model
        self.event_adapt = [list(range(dro_model.num_scen))]
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
            raise RuntimeError('The model is unsolved.')

        solution = dro_model.solution
        if np.isnan(solution.objval):
            msg = 'No solution available. '
            msg += f'{solution.solver} solution status: {solution.status}.'
            raise RuntimeError(msg)

        var_sol = dro_model.ro_model.rc_model.vars[1].get()
        edict = event_dict(self.event_adapt)
        if rvar is None:
            outputs = []
            for eindex in range(len(self.event_adapt)):
                indices = (self.ro_first + eindex*self.size +
                           np.arange(self.size, dtype=int))
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

                coeff = np.array([np.nan] * sp.shape[0])
                coeff[sol_indices] = sol_vec

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

    def __call__(self, *args):

        return self.to_affine()(*args)


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

    def __call__(self, *args):

        return self.to_affine()(*args)


class RandVar(Vars):
    """
    The RandVar class creates an object of random variable array.
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

    def assign(self, values, sw=False):

        if not isinstance(values, (np.ndarray, Real)):
            raise TypeError('The second argument does not provide numerical values.')

        if not sw:
            values = np.array(values, dtype=float) + np.zeros(self.shape, dtype=float)
            values = values.reshape(self.shape)
        else:
            if isinstance(values, pd.Series):
                values = values.values

            value_list = []
            for i in range(self.model.top.num_scen):
                value = np.array(values[i], dtype=float)
                value += np.zeros(self.shape, dtype=float)
                value_list.append(value)
            values = pd.Series(value_list, index=self.model.top.series_scen.index)

        return RandVal(self, values, sw)


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

    def norm(self, degree, method=None):
        """
        Return the first, second, or infinity norm of a 1-D array.

        Refer to `rsome.norm` for full documentation.

        See Also
        --------
        rso.norm : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().norm(degree, method)

        return DecConvex(expr, self.event_adapt)

    def pnorm(self, degree, method=None):
        """
        Return p-norm of a 1-D array, where p is a real number
        larger than 1.

        Refer to `rsome.math.pnorm` for full documentation

        See Also
        --------
        rsome.math.pnorm : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().pnorm(degree, method)

        return DecConvex(expr, self.event_adapt)

    def square(self):
        """
        Return the element-wise square of an array.

        Refer to `rsome.square` for full documentation.

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

    def quad(self, qmat):
        """
        Return the quadratic expression var @ qmat @ var.

        Refer to `rsome.quad` for full documentation.

        See Also
        --------
        rso.quad : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().quad(qmat)

        return DecConvex(expr, self.event_adapt)

    def rsocone(self, y, z):
        """
        Return the rotated second-order cone constraint.

        Refer to `rsome.rsocone` for full documentation.

        See Also
        --------
        rso.rsocone : equivalent function
        """

        if not self.fixed:
            raise ValueError('Incorrect convex expressions.')

        expr = super().rsocone(y, z)

        return DecConvex(expr, self.event_adapt)

    def sum(self, axis=None):

        expr = super().sum(axis)

        return DecAffine(self.dro_model, expr, self.event_adapt, self.fixed)

    def trace(self):
        """
        Return the trace of a 2D array.

        Refer to `rsome.lp.trace` for full documentation.

        See Also
        --------
        rsome.lp.rsocone : equivalent function
        """

        expr = super().trace()

        return DecAffine(expr.dro_model, expr, self.event_adapt, self.fixed)

    def expcone(self, x, z):
        """
        Return the exponential cone constraint z*exp(x/z) <= affine.

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
        Return the natural exponential function exp(affine).

        Refer to `rsome.exp` for full documentation.

        See Also
        --------
        rso.exp : equivalent function
        """

        return DecConvex(Convex(self, np.zeros(self.shape), 'X', 1),
                         self.event_adapt)

    def power(self, p, q=1):
        """
        Return the element-wise integer power of the given affine
        array, i.e. affine ** (p/q)

        Refer to `rsome.power` for full documentation.

        See Also
        --------
        rso.power : equivalent function
        """

        expr = super().power(p, q)

        return DecConvex(expr, self.event_adapt)

    def gmean(self, beta=None):
        """
        Return the weighted geometric mean of a 1-D array. The weights
        are specified by an array-like structure beta. It is expressed
        as prod(affine ** beta) ** (1/sum(beta))

        Refer to `rsome.gmean` for full documentation.

        See Also
        --------
        rso.gmean : equivalent function
        """

        expr = super().gmean(beta)

        return DecConvex(expr, self.event_adapt)

    def pexp(self, scale):
        """
        Return the perspective of natural exponential function
        scale * exp(affine/scale).

        Refer to `rsome.pexp` for full documentation.

        See Also
        --------
        rso.pexp : equivalent function
        """

        return DecPerspConvex(PerspConvex(self, scale,
                                          np.zeros(self.shape), 'X', 1),
                              self.event_adapt)

    def log(self):
        """
        Return the natural logarithm function log(affine).

        Refer to `rsome.log` for full documentation.

        See Also
        --------
        rso.log : equivalent function
        """

        return DecConvex(Convex(self, np.zeros(self.shape), 'L', -1),
                         self.event_adapt)

    def logdet(self):
        """
        Return the log-determinant of a positive semidefinite matrix
        expressed as a two-dimensional array.

        Refer to `rsome.logdet` for full documentation.

        See Also
        --------
        rso.logdet : equivalent function
        """

        expr = super().logdet()

        return DecConvex(expr, self.event_adapt)

    def rootdet(self):
        """
        Return the root-determinant of a positive semidefinite matrix
        expressed as a two-dimensional array. The root-determinant is
        expressed as (det(A))**(1/L), where L is the dimension of the
        two-dimensinoal array.

        Refer to `rsome.rootdet` for full documentation.

        See Also
        --------
        rso.rootdet : equivalent function
        """

        expr = super().rootdet()

        return DecConvex(expr, self.event_adapt)

    def plog(self, scale):
        """
        Return the perspective of natural logarithm function
        scale * log(affine/scale).

        Refer to `rsome.plog` for full documentation.

        See Also
        --------
        rso.plog : equivalent function
        """

        return DecPerspConvex(PerspConvex(self, scale,
                                          np.zeros(self.shape), 'L', -1),
                              self.event_adapt)

    def softplus(self):
        """
        Return the softplus function log(1 + exp(var)).

        Refer to `rsome.softplus` for full documentation.

        See Also
        --------
        rso.softplus : equivalent function
        """

        return DecConvex(Convex(self, np.zeros(self.shape), 'F', 1),
                         self.event_adapt)

    def entropy(self):
        """
        Return the natural exponential function -sum(affine*log(affine)).

        Refer to `rsome.entropy` for full documentation.

        See Also
        --------
        rso.entropy : equivalent function
        """

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
        elif isinstance(left, ExpPiecewiseConvex):
            pieces = [piece <= 0 for piece in left.pieces]
            return ExpPWConstr(left.model, pieces)
        elif isinstance(left, PiecewiseConvex):
            pieces = [piece <= 0 for piece in left.pieces]
            return PWConstr(left.model, pieces)

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
        elif isinstance(left, ExpPiecewiseConvex):
            pieces = [piece <= 0 for piece in left.pieces]
            return ExpPWConstr(left.model, pieces)
        elif isinstance(left, PiecewiseConvex):
            pieces = [piece <= 0 for piece in left.pieces]
            return PWConstr(left.model, pieces)

    def __eq__(self, other):

        left = self - other
        if isinstance(left, DecAffine):
            return DecLinConstr(left.model, left.linear, -left.const,
                                np.ones(left.size), left.event_adapt,
                                left.fixed, left.ctype)
        elif isinstance(left, DecRoAffine):
            return DecRoConstr(left, 1, left.event_adapt, left.ctype)

    def __rshift__(self, other):

        left = self - other
        if isinstance(left, DecAffine):
            constr = super().__rshift__(other)
            return DecLMIConstr(constr, left.event_adapt)
        else:
            msg = 'Linear matrix inequalities only apply to affine expressions.'
            raise TypeError(msg)

    def __lshift__(self, other):

        return (-self).__rshift__(-other)

    @property
    def E(self):

        affine = Affine(self.model, self.linear, self.const)
        return DecAffine(self.dro_model, affine, fixed=self.fixed, ctype='E')

    def __call__(self, *args):

        if self.model.mtype != 'V':
            raise ValueError('Unsupported affine expression.')

        if self.dro_model.solution is None:
            raise SyntaxError('No available solution!')
        else:
            decs = self.dro_model.rule_var()
            ns = len(decs)
            values = []
            for i, dec in zip(self.dro_model.series_scen.index, decs):
                if isinstance(dec, Affine):
                    xs = dec()
                elif isinstance(dec, RoAffine):
                    args_sw = [RandVal(arg.rvar, arg.values.loc[i]) if arg.sw else arg
                               for arg in args]
                    xs = dec(*args_sw)

                nvar = self.linear.shape[1]
                output = (self.linear @ xs[:nvar]).reshape(self.shape)
                output += self.const.reshape(self.shape)

                if output.ndim == 0:
                    output = output.item()

                values.append(output)

            if ns > 1 and len(self.event_adapt) > 1:
                return pd.Series(values, index=self.dro_model.series_scen.index)
            else:
                return values[0]


class DecConvex(Convex):
    """
    The DecConvex class creates an object of convex functions
    of decision variables.
    """

    def __init__(self, convex, event_adapt):

        super().__init__(convex.affine_in, convex.affine_out,
                         convex.xtype, convex.sign, convex.multiplier,
                         params=convex.params)
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

    def __call__(self):

        if self.model.mtype != 'V':
            raise ValueError('Unsupported affine expression.')

        if self.model.top.solution is None:
            raise SyntaxError('No available solution!')
        else:
            values_in = self.affine_in()
            if isinstance(self.affine_out, Affine):
                values_out = self.affine_out()
            else:
                values_out = self.affine_out
            if not isinstance(values_in, pd.Series):
                values_in = pd.Series([values_in])
            if not isinstance(values_out, pd.Series):
                values_out = pd.Series([values_out] * len(values_in))

            output = []
            for value_in, value_out in zip(values_in, values_out):
                if self.xtype == 'A':
                    output.append(self.multiplier*self.sign*abs(value_in) + value_out)
                elif self.xtype == 'M':
                    item = self.multiplier*self.sign*abs(value_in).sum()
                    item += value_out
                    output.append(item)
                elif self.xtype == 'N':
                    d = self.params
                    if isinstance(d, Iterable):
                        d = d[0] / d[1]
                    item = self.multiplier*self.sign*np.linalg.norm(abs(value_in), d)
                    item += value_out
                    output.append(item)
                elif self.xtype == 'E':
                    item = self.multiplier*self.sign*((value_in**2).sum())**0.5
                    item += value_out
                    output.append(item)
                elif self.xtype == 'I':
                    item = self.multiplier*self.sign*abs(value_in).max()
                    item += value_out
                    output.append(item)
                elif self.xtype == 'S':
                    output.append(self.multiplier**2*self.sign*(value_in**2) + value_out)
                elif self.xtype == 'Q':
                    item = self.multiplier**2*self.sign*(value_in**2).sum()
                    item += value_out
                    output.append(item)
                elif self.xtype == 'X':
                    output.append(self.multiplier*self.sign*np.exp(value_in) + value_out)
                elif self.xtype == 'L':
                    item = -self.multiplier*self.sign*np.log(value_in)
                    item += value_out
                    output.append(item)
                elif self.xtype == 'P':
                    item = self.multiplier*self.sign*(value_in*np.log(1/value_in)).sum()
                    item += value_out
                    output.append(item)
                elif self.xtype == 'G':
                    d = self.params
                    if isinstance(d, Iterable):
                        d = d[0] / d[1]
                    item = self.multiplier*self.sign*np.linalg.norm(abs(value_in), d)
                    item += value_out
                    output.append(item)
                else:
                    raise ValueError('Unsupported convex/concave expression.')

            if len(output) > 1 and len(self.event_adapt) > 1:
                output = pd.Series(output, index=values_in.index)
            else:
                output = output[0]

            return output


class ExpPiecewiseConvex(PiecewiseConvex):
    """
    The ExpPiecewiseConvex class creates an object of the expectation
    of a piecewise function.
    """

    def __init__(self, model, pieces, sign=1, add_sign=1):

        expect_pieces = []
        for piece in pieces:
            if isinstance(piece, (DecVar, DecVarSub)):
                piece = piece.to_affine()
            if isinstance(piece, (RandVar, RandVarSub)):
                piece = piece.rand_to_roaffine(model.vt_model)
            if isinstance(piece, (DecAffine, DecRoAffine)):
                piece.ctype = 'E'

            expect_pieces.append(piece)

        super().__init__(model, expect_pieces, sign, add_sign)

    def __add__(self, other):

        if isinstance(other, (DecVar, DecVarSub)):
            other = other.to_affine()

        if isinstance(other, DecAffine):
            if not other.fixed and other.ctype != 'E':
                raise ValueError('Incorrect expectation expressions.')
        elif isinstance(other, DecRoAffine):
            if other.ctype != 'E':
                raise ValueError('Incorrect expectation expressions.')
        elif not isinstance(other, (Real, np.ndarray)):
            raise TypeError('Unsupported expectation expressions.')

        piecewise = super().__add__(other)

        return ExpPiecewiseConvex(piecewise.model, piecewise.pieces,
                                  piecewise.sign, piecewise.add_sign)

    def __neg__(self):

        return ExpPiecewiseConvex(self.model, self.pieces, -self.sign, self.add_sign)

    def __sub__(self, other):

        return self.__add__(-other)

    def __rsub__(self, other):

        return (-self).__add__(other)

    def __radd__(self, other):

        return self.__add__(other)

    def __mul__(self, other):

        piecewise = super().__mul__(other)

        return ExpPiecewiseConvex(piecewise.model, piecewise.pieces,
                                  piecewise.sign, piecewise.add_sign)

    def __rmul__(self, other):

        return self.__mul__(other)

    def __le__(self, other):

        if isinstance(other, (np.ndarray, Real)):
            constr = super().__le__(other)
        else:
            left = self - other
            constr = left.__le__(0)

        return ExpPWConstr(constr.model, constr.pieces)

    def __ge__(self, other):

        right = other - self

        return right.__le__(0)


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
    """
    The DecRoaffine class creats an object of uncertain affine functions
    for scenario-wise DRO models.
    """

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
                    cond1 = self.ctype == 'E'
                    cond2 = not other.fixed or len(other.event_adapt) > 1
                    cond3 = other.ctype == 'E'
                    if (cond1 and cond2) or cond3:
                        raise TypeError('Incorrect expectation expressions.')
                other = other.to_affine()
            event_adapt = comb_set(self.event_adapt, other.event_adapt)
            ctype = 'E' if 'E' in self.ctype + other.ctype else 'R'
        elif (isinstance(other, (Real, np.ndarray, RoAffine)) or
              sp.issparse(other)):
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

    def __call__(self, *args):

        nrand = self.rand_model.last
        nscen = self.dec_model.top.num_scen
        # rvec = np.zeros(nrand)
        rvecs = pd.DataFrame(np.zeros((nscen, nrand)),
                             index=self.dec_model.top.series_scen.index)
        sw = False
        for arg in args:
            if not isinstance(arg, RandVal):
                raise TypeError('Unsupported type for defining random variable values.')

            index = range(arg.rvar.first, arg.rvar.last)
            # rvec[index] = arg.values.ravel()
            if not arg.sw:
                rvecs.loc[:, index] = arg.values.ravel()
            else:
                sw = True
                for i in arg.values.index:
                    rvecs.loc[i, index] = arg.values.loc[i].ravel()

        raffine_values = self.raffine()
        affine_values = self.affine()

        if isinstance(raffine_values, pd.Series) or sw:
            output = []
            for i in rvecs.index:
                if isinstance(raffine_values, pd.Series):
                    raffine_value = raffine_values.loc[i]
                    affine_value = affine_values.loc[i]
                else:
                    raffine_value = raffine_values
                    affine_value = affine_values
                nrand = raffine_value.shape[1]

                item = (raffine_value@rvecs.loc[i].values[:nrand]).reshape(self.shape)
                item += affine_value
                output.append(item)
            output = pd.Series(output, index=rvecs.index)
        else:
            nrand = raffine_values.shape[1]
            output = (raffine_values@rvecs.iloc[0].values[:nrand]).reshape(self.shape)
            output += affine_values

        return output


class DecLinConstr(LinConstr):
    """
    The DecLinConstr class creats an object of linear constraints of
    generic decision variables.
    """

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

    def forall(self, ambset):

        self.ambset = ambset

        return self


class DecBounds(Bounds):

    def __init__(self, bounds, event_adapt=None):

        super().__init__(bounds.model, bounds.indices, bounds.values,
                         bounds.btype)
        self.event_adapt = event_adapt


class DecCvxConstr(CvxConstr):

    def __init__(self, constr, event_adapt):

        super().__init__(constr.model, constr.affine_in,
                         constr.affine_out, constr.multiplier, constr.xtype,
                         params=constr.params)
        self.event_adapt = event_adapt


class ExpPWConstr(PWConstr):

    def __init__(self, model, pieces, ambset=None):

        super().__init__(model, pieces)
        self.ambset = ambset

    def forall(self, ambset):

        return ExpPWConstr(self.model, self.pieces, ambset)


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


class DecLMIConstr(LMIConstr):

    def __init__(self, constr, event_adapt, ctype='R'):

        super().__init__(constr.model, constr.linear, constr.const, constr.dim)
        self.event_adapt = event_adapt
        self.ctype = ctype


class DecRule:
    """
    The DecRule class creats an object of decision rules for RO models.
    """

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
            raise RuntimeError('The model is unsolved. ')

        solution = self.model.solution
        if np.isnan(solution.objval):
            msg = 'No solution available. '
            msg += f'{solution.solver} solution status: {solution.status}.'
            raise RuntimeError(msg)

        if rvar is None:
            return self.fixed.get()
        else:
            if rvar.model.mtype != 'S':
                raise ValueError('The input is not a random variable.')
            ldr_row, ldr_col = self.size, self.model.rc_model.vars[-1].last
            ldr_coeff = np.array([[np.nan] * ldr_col] * ldr_row)
            rand_ind = rvar.get_ind()
            row_ind, col_ind = np.where(self.depend == 1)
            ldr_coeff[row_ind, col_ind] = self.var_coeff.get()

            rv_shape = rvar.to_affine().shape
            return ldr_coeff[:, rand_ind].reshape(self.shape + rv_shape)

    def __call__(self, *args):

        return self.to_affine()(*args)


class DecRuleSub:
    """
    The DecRuleSub class creats an object of a subset of decision rule
    for RO models.
    """

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

    def __call__(self, *args):

        return self.to_affine()(*args)


class LinProg:
    """
    The LinProg class creates an object of linear program.
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
                            for i, coeff in enumerate(self.obj) if coeff])
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
            string += ' <= ' if self.sense[i] == 0 else ' = '
            string += '{}\n'.format(self.const[i])

        ub, lb = self.ub, self.lb
        nvar = len(ub)
        string += 'Bounds\n'
        for i in range(nvar):
            string += '{} <= x{} <= {}\n'.format(lb[i], i+1, ub[i])

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
    """
    The Solution class creats an object summarizing solution information.
    """

    def __init__(self, solver, objval, x, status, time, xs=None, y=None):

        self.solver = solver
        self.objval = objval
        self.x = x
        self.xs = xs
        self.y = y
        self.status = status
        self.time = time

    def __repr__(self):

        msg = f"Solver:          {self.solver}\n"
        msg += f"Solution status: {self.status}\n"
        msg += f"Solution time:   {self.time:.4f}s\n"
        msg += "-------------------------------------\n"
        msg += f"Objective value: {self.objval}\n"
        primal_available = "Available" if self.x is not None else "Unavailable"
        dual_available = "Available" if self.y is not None else "Unavailable"
        msg += f"Primal solution: {primal_available}\n"
        msg += f"Dual solution:   {dual_available}"

        return msg


class Scen:
    """
    The Scen class creats an object of scenarios for the ambiguity set and
    scenario-wise decision adaptive rules.
    """

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
            if not isinstance(arg, (LinConstr, CvxConstr, Bounds, ConeConstr, LMIConstr)):
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


class RandVal:
    """
    The RandVal class creats an object that pairs the random variable and its
    observed value.
    """

    def __init__(self, rvar, values, sw=False):

        self.rvar = rvar
        self.values = values
        self.sw = sw

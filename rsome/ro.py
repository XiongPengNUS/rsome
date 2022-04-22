# from .socp import Model as SOCModel
from .gcp import Model as GCPModel
from .lp import LinConstr, Bounds, CvxConstr, ConeConstr, ExpConstr, KLConstr
from .lp import Vars, VarSub, Affine, Convex
from .lp import DecRule
from .lp import RoAffine, RoConstr
from .lp import Solution, def_sol
import numpy as np
from numbers import Real
from collections.abc import Iterable
# from .lpg_solver import solve as def_sol


class Model:
    """
    The Model class creates an object of robust optimization models
    """

    def __init__(self, name=None):

        self.rc_model = GCPModel(mtype='R', top=self)
        self.sup_model = GCPModel(nobj=True, mtype='S', top=self)

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

    def dvar(self, shape=(), vtype='C', name=None, aux=False):
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
        new_var : rsome.lp.Vars
            An array of new random variables
        """

        new_var = self.sup_model.dvar(shape, 'C', name)
        return new_var

    def ldr(self, shape=(), name=None):

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
        """
        Minimize the given objective function.

        Parameters
        ----------
        obj : RSOME expression, numeric constant
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
        obj : RSOME expression, numeric constant
            The objective function

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

    def minmax(self, obj, *args):

        """
        Minimize the maximum objective value over the given uncertainty set.

        Parameters
        ----------
        obj : RSOME expression, numeric constant
            Objective function involving random variables
        *args
            Constraints or collections of constraints of random variables
            used for defining the uncertainty set

        Notes
        -----
        The uncertainty set defined for the objective function is considered
        the default uncertainty set for the robust model.
        """

        if self.obj is not None:
            raise SyntaxError('Redefinition of the objective is not allowed.')

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
                raise ValueError('Models mismatch.')
            sup_model.st(item)

        self.obj = obj
        self.obj_support = sup_model.do_math(primal=False, obj=False)
        self.sign = 1
        self.pupdate = True
        self.dupdate = True

    def maxmin(self, obj, *args):

        """
        Maximize the minimum objective value over the given uncertainty set.

        Parameters
        ----------
        obj : RSOME expression, numeric constant
            Objective function involving random variables
        *args
            Constraints or collections of constraints of random variables
            used for defining the uncertainty set

        Notes
        -----
        The uncertainty set defined for the objective function is considered
        the default uncertainty set for the robust model.
        """

        if self.obj is not None:
            raise SyntaxError('Redefinition of the objective is not allowed.')

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
                raise ValueError('Models mismatch.')
            sup_model.st(item)

        self.obj = obj
        self.obj_support = sup_model.do_math(primal=False, obj=False)
        self.sign = - 1
        self.pupdate = True
        self.dupdate = True

    def st(self, *arg):
        """
        Define constraints that an optimization model subject to.

        Parameters
        ----------
        *args : RSOME constraints, iterables
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

            elif isinstance(constr, (LinConstr, Bounds, CvxConstr,
                                     ConeConstr, ExpConstr, KLConstr)):
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
                    left_constr = RoConstr(left, sense=0)
                    left_constr.support = constr.support
                    right_constr = RoConstr(right, sense=0)
                    right_constr.support = constr.support
                    self.all_constr.append(left_constr)
                    self.all_constr.append(right_constr)
            else:
                raise TypeError('Unknown type of constraints')

        self.pupdate = True
        self.dupdate = True

    def do_math(self, primal=True):
        """
        Return the linear, second-order cone, or exponential cone
        programming problem as the standard formula or deterministic
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
                return self.rc_model.do_math(False, obj=True)

        self.rc_model.reset()
        if isinstance(self.obj, (Vars, VarSub, Affine, Convex, Real)):
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
            if isinstance(constr, (LinConstr, Bounds, CvxConstr,
                                   ExpConstr, KLConstr)):
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
                So far the argument only applies to Gurobi and MOSEK.
        """

        if solver is None:
            solution = def_sol(self.do_math(), display, params)
        else:
            solution = solver.solve(self.do_math(), display, params)

        if isinstance(solution, Solution):
            self.rc_model.solution = solution
        else:
            self.rc_model.solution = None

        self.solution = self.rc_model.solution

    def get(self):
        """
        Return the optimal objective value of the solved model.

        Notes
        -----
        An error message is given if the model is unsolved or no solution
        is obtained.
        """

        if self.rc_model.solution is None:
            raise RuntimeError('The model is unsolved or no solution is obtained.')
        return self.sign * self.rc_model.solution.objval

    def optimal(self):

        return self.solution is not None

from .socp import Model as SOCModel
from .lp import LinConstr, Bounds, CvxConstr, ConeConstr
from .lp import Vars, VarSub, Affine, Convex
from .lp import DecRule, DecRuleSub
from .lp import RoAffine, RoConstr
from .lp import Solution
from .subroutines import *
import numpy as np
from scipy.sparse import csr_matrix
from collections import Iterable
from .lpg_solver import solve as def_sol


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

            elif isinstance(constr, (LinConstr, Bounds, CvxConstr, ConeConstr)):
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
            else:
                self.do_math(primal=True)
                return self.rc_model.do_math(False, obj=True)

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

    def solve(self, solver=None, display=True, export=False, params={}):
        """
        Solve the model with the selected solver interface.

        Parameters
        ----------
            solver : {None, lpg_solver, grb_solver, msk_solver}
                Solver interface used for model solution. Use default solver
                if solver=None.
            display : bool
                Display option of the solver interface.
            export : bool
                Export option of the solver interface. A standard model file
                is generated if the option is True.
            params : dict
                A dictionary that specifies parameters of the selected solver.
                So far the argument only applies to Gurobi and MOSEK.
        """

        if solver is None:
            solution = def_sol(self.do_math(), display, export, params)
        else:
            solution = solver.solve(self.do_math(), display, export, params)

        if isinstance(solution, Solution):
            self.rc_model.solution = solution
        else:
            if solution is None:
                self.rc_model.solution = None
            else:
                x = solution.x
                self.rc_model.solution = Solution(x[0], x, solution.status)

        self.solution = self.rc_model.solution

    def get(self):

        if self.rc_model.solution is None:
            raise SyntaxError('The model is unsolved or no feasible solution.')
        return self.sign * self.rc_model.solution.objval

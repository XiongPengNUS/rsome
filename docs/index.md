<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Introduction

RSOME (Robust Stochastic Optimization Made Easy) is an open-source Python package for modeling generic optimization problems. Models in RSOME are constructed by variables, constraints, and expressions that are formatted as N-dimensional arrays. These arrays are consistent with the NumPy library in terms of syntax and operations, including broadcasting, indexing, slicing, element-wise operations, and matrix calculation rules, among others. In short, RSOME provides a convenient platform to facilitate developments of optimization models and their applications.


## Installing RSOME and Solvers

The RSOME package can be installed with the `pip` command:

***

`pip install rsome`

***

The current version of RSOME supports deterministic, robust optimization and distributionally robust optimization problems. In the default configuration, RSOME relies on open-source solvers imported from the `scipy.optimize` package to solve linear programming problems. If SciPy 1.9.0 or above is installed, the default solver `milp()` is capable of solving problems with integer variables. However, if the installed SciPy package is 1.8.1 or below, the default solver is `linprog()` and it is only able to address continuous variables. Besides the default solver, RSOME also provides interfaces for other open-source and commercial solvers. Detailed information of these solver interfaces is presented in the following table. 

| Solver | License  type | Required version | RSOME interface |Integrality constraints| Second-order cone constraints| Exponential cone constraints
|:-------|:--------------|:-----------------|:----------------|:------------------------|:---------------------|:--------------|
|[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)| Open-source | >= 1.2.1 | `lpg_solver` | Yes for 1.9.0 or above | No | No |
|[CyLP](https://github.com/coin-or/cylp)| Open-source | >= 0.9.0 | `clp_solver` | Yes | No | No |
|[OR-Tools](https://developers.google.com/optimization/install) | Open-source | >= 7.5.7466 | `ort_solver` | Yes | No | No |
|[ECOS](https://github.com/embotech/ecos-python) | Open-source | >= 2.0.10 | `eco_solver` | Yes | Yes | Yes |
|[Gurobi](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html)| Commercial | >= 9.1.0 | `grb_solver` | Yes | Yes | No |
|[MOSEK](https://docs.mosek.com/9.2/pythonapi/install-interface.html) | Commercial | >= 9.1.11 | `msk_solver` | Yes | Yes | Yes |
|[CPLEX](https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) | Commercial | >= 12.9.0.0 | `cpx_solver` | Yes | Yes | No |
|[COPT](https://www.shanshu.ai/copt) | Commercial | >= 5.0.1 | `cpt_solver` | Yes | Yes | No |


## A Linear Programming Example

The RSOME package supports specifying models using highly readable algebraic expressions that are consistent with NumPy syntax. A very simple linear program example is provided below,

$$
\begin{align}
\max ~&3x + 4y \\
\text{s.t.}~&2.5x + y \leq 20 \\
&5x + 3y \leq 30 \\
&x + 2y \leq 16 \\
&|y| \leq 2,
\end{align}
$$

and it is used to illustrate the steps of solving optimization models.


```python
from rsome import ro                # import the ro modeling tool

model = ro.Model('LP model')        # create a Model object
x = model.dvar()                    # define a decision variable x
y = model.dvar()                    # define a decision variable y

model.max(3*x + 4*y)                # maximize the objective function
model.st(2.5*x + y <= 20)           # specify the 1st constraints
model.st(5*x + 3*y <= 30)           # specify the 2nd constraints
model.st(x + 2*y <= 16)             # specify the 3rd constraints
model.st(abs(y) <= 2)               # specify the 4th constraints

model.solve()                       # solve the model by the default solver
```

    Being solved by the default LP solver...
    Solution status: 0
    Running time: 0.0426s


In this sample code, a model object is created by calling the constructor `Model()` imported from the `rsome.ro` toolbox. Based on the model object, decision variables `x` and `y` are created by the method `dvar()`. These variables are then used in specifying the objective function and model constraints. The last step is calling the `solve()` method to solve the problem. Once the solution completes, a message showing the solution status and running time will be printed.

You may find the interpretation of the solution status code of `linprog()` from the website [`scipy.optimize.linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html). The status code `0` suggests that the problem was solved to optimality (subject to tolerances), and an optimal solution is available. The optimal solution and the corresponding objective value can be attained by the `get()` method.


```python
print('x: {:.3f}'.format(x.get()))
print('y: {:.3f}'.format(y.get()))
print('Objective: {:.3f}'.format(model.get()))
```

    x: 4.800
    y: 2.000
    Objective: 22.400


The example above shows that RSOME models can be formulated via straightforward and highly readable algebraic expressions, and the formulated model can be transformed into a standard form, which is then solved by the integrated solver. The basic information of the standard form can be retrieved by calling the `do_math()` method of the RSOME model object.


```python
formula = model.do_math()
print(formula)
```

```
    Conic program object:
    =============================================
    Number of variables:           3
    Continuous/binaries/integers:  3/0/0
    ---------------------------------------------
    Number of linear constraints:  6
    Inequalities/equalities:       6/0
    Number of coefficients:        11
    ---------------------------------------------
    Number of SOC constraints:     0
    ---------------------------------------------
    Number of ExpCone constraints: 0
    
```

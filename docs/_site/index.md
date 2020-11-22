<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Introduction

RSOME (Robust Stochastic Optimization Made Easy) is an open-source Python package for generic modeling optimization problems. Models in RSOME are constructed by variables, constraints, and expressions that are formatted as N-dimensional arrays. These arrays are consistent with the NumPy library in terms of syntax and operations, including broadcasting, indexing, slicing, element-wise operations, and matrix calculation rules, among others. In short, RSOME provides a convenient platform to facilitate developments of optimization models and their applications.

The current version of RSOME supports models that fit the state-of-the-art robust stochastic optimization framework, proposed in the paper [robust stochastic optimization (RSO) framework](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603?af=R). The default solver of RSOME is the linear programming (LP) solver `linprog()` imported from the `scipy.optimize` package. We also provide interfaces with commercial solvers like [Gurobi](https://www.gurobi.com/) and [MOSEK](https://www.mosek.com/) so that we can solve problems with integer variables and second-order cone constraints.  


## Installing RSOME and solvers

The RSOME package can be installed with the <code>pip</code> command:

***

`pip install rsome`

***

For the current version, the `scipy.optimize.linprog() ` function is used as a default solver for LP problems. Interfaces with other solvers like Gurobi and MOSEK are also integrated in RSOME, and you may follow [Installing the Anaconda Python distribution](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html) or [Installation - MOSEK optimizer API for Python](https://docs.mosek.com/9.2/pythonapi/install-interface.html) to complete the solver installation.


## A linear programming example

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
from rsome import ro                # Import the ro modeling tool

model = ro.Model('LP model')        # Create a Model object
x = model.dvar()                    # Define a decision variable x
y = model.dvar()                    # Define a decision variable y

model.max(3*x + 4*y)                # Maximize the objective function
model.st(2.5*x + y <= 20)           # Specify the 1st constraints
model.st(5*x + 3*y <= 30)           # Specify the 2nd constraints
model.st(x + 2*y <= 16)             # Specify the 3rd constraints
model.st(abs(y) <= 2)               # Specify the 4th constraints

model.solve()                       # Solve the model by the default solver
```

    Being solved by the default LP solver...
    Solution status: 0
    Running time: 0.0426s


In this sample code, a model object is created by calling the constructor <code>Model()</code> imported from the <code>rsome.ro</code> toolbox. Based on the model object, decision variables <code>x</code> and <code>y</code> are created by the method <code>dvar()</code>. These variables are then used in specifying the objective function and model constraints. The last step is calling the <code>solve()</code> method to solve the problem via the imported solver interface <code>grb</code>. Once the solution completes, a message showing the solution status and running time will be printed.

You may find the interpretation of the solution status code of `linprog()` from the website [`scipy.optimize.linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html). The status code <code>0</code> suggests that the problem was solved to optimality (subject to tolerances), and an optimal solution is available. The optimal solution and the corresponding objective value can be attained by the <code>get()</code> method.


```python
print('x:', x.get())
print('y:', y.get())
print('Objective:', round(model.get(), 2))
```

    x: [4.8]
    y: [2.]
    Objective: 22.4


The example above shows that RSOME models can be formulated via straightforward and highly readable algebraic expressions, and the formulated model can be transformed into a standard form, which is then solved by the Gurobi (or MOSEK) solver. The basic information of the standard form can be retrieved by calling the <code>do_math()</code> method of the RSOME model object.


```python
formula = model.do_math()
print(formula)
```

    Second order cone program object:
    =============================================
    Number of variables:          3
    Continuous/binaries/integers: 3/0/0
    ---------------------------------------------
    Number of linear constraints: 6
    Inequalities/equalities:      6/0
    Number of coefficients:       11
    ---------------------------------------------
    Number of SOC constraints:    0

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Getting Started

RSOME is an open-source algebraic library for modeling generic optimization problems under uncertainty. It provides highly readable and mathematically intuitive modeling environment based on the state-of-the-art robust stochastic optimization framework.

This guide introduces the main components, basic data structures, and syntax rules of the RSOME package. For installations, please refer to our [Home Page](index) for more information.

## Modeling Environments <a name="section1.1"></a>

The RSOME package provides the following two modules for formulating optimization problems under uncertainty:

- The `ro` module is a tailored modeling framework for robust optimization problems. This module provides modeling tools designed specifically for constructing uncertainty sets and specifying affine decision rules in multi-stage decision-making applications. 

- The `dro` module is built upon the distributionally robust optimization framework proposed in [Chen et al. (2020)](#ref1). Modeling tools are provided for constructing event-wise ambiguity sets and specifying event-wise adaptation policies.

These two modeling frameworks follow consistent syntax in defining variables, objective functions, and constraints. The only differences are in specifying recourse adaptations and uncertainty/ambiguity sets. Notice that the `dro` module is a more general modeling framework, since a classic robust optimization problem can be treated as a special case of distributionally robust optimization, where the ambiguity set, specifying only the support information, reduces to an uncertainty set. The `ro` module is less general but the toolkit enables users to formulate uncertainty sets and decision adaptations in a more concise and intuitive manner. 

In this section, we will use the `ro` module as a general modeling environment for deterministic problems. Guidelines of robust and distributionally robust optimization problems are presented in [RSOME for robust optimization](ro_rsome) and [RSOME for distributionally robust optimization](dro_rsome), respectively.

## Introduction to the `rsome.ro` Environment <a name="section1.2"></a>

### Models

In RSOME, all optimization models are specified based on a `Model` type object. Such an object is created by the constructor `Model()` imported from the `rsome.ro` modeling environment.


```python
from rsome import ro            # import the ro modeling tool

model = ro.Model('My model')    # create a Model object
```

The code above defines a new `Model` object `model`, with the name specified to be `'My model'`. You could also leave the name unspecified and the default name is `None`.

### Decision Variables and Linear Constraints

Decision variables of a model can be defined by the method `dvar()`.
```
dvar(self, shape=(), vtype='C', name=None, aux=False)
    Returns an array of decision variables with the given shape
    and variable type.
    
    Parameters
    ----------
    shape : int or tuple
        Shape of the variable array. The variable is a scalar if
        the shape is unspecified.
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
```

Variables in RSOME can be formatted as N-dimensional arrays, which are consistent with the widely used NumPy arrays in the definition of dimensions, shapes, and sizes. Users could use the `dvar()` method of the `Model` object to specify the shape and type (`'C'` for continuous, `'B'` for binary, and `'I'` for general integer) of the decision variable array, as shown by the following examples.

```python
x = model.dvar(3, vtype='I')    # 3 integer variables as a 1D array
y = model.dvar((3, 5), 'B')     # 3x5 binary variables as a 2D array
z = model.dvar((2, 3, 4, 5))    # 2x3x4x5 continuous variables as a 4D array
```

RSOME variables are also compatible with the standard NumPy array operations, such as element-wise computation, matrix calculation rules, broadcasting, indexing and slicing, etc. It thus enables users to define blocks of constraints with the array system. For example, the constraint system

$$
\begin{align}
&\sum\limits_{i\in[I]}b_ix_i = 1 && \\
&\sum\limits_{i\in[I]}A_{ji}x_i \leq c_j && j\in[J] \\
&\sum\limits_{j\in[J]}\sum\limits_{i\in I}y_{ji} \geq 1 &&\\
&\sum\limits_{i\in[I]}y_{ji} \geq 0 && j\in [J] \\
&A_{ji}x_i \geq 1 &&\forall j\in[J], i\in[I] \\
&A_{ji}y_{ji} + x_i \geq 0 && \forall j\in [J], i\in[I]
\end{align}
$$

with decision variable \\(\pmb{x}\in\mathbb{R}^I\\) and \\(\pmb{y}\in\mathbb{R}^{J\times I}\\), as well as parameters \\(\pmb{A}\in\mathbb{R}^{J\times I}\\), \\(\pmb{b}\in\mathbb{R}^I\\), and \\(\pmb{c}\in\mathbb{R}^J\\), can be conveniently specified by the code segment below.

```python
x = model.dvar(I)               # define x as a 1D array of I variables
y = model.dvar((J, I))          # define y as a 2D array of JxI variables

b @ x == 1                      
A @ x <= c
y.sum() >= 1
y.sum(axis=1) >= 0
A * x >= 1
A*y + x >= 0
```

RSOME arrays can also be used in specifying the objective function of the optimization model. Note that the objective function must be one affine expression. In other words, the `size` attribute of the expression must be one, otherwise an error message would be generated.


```python
model.min(b @ x)        # minimize the objective function b @ x
model.max(b @ x)        # maximize the objective function b @ x
```

Model constraints can be specified by the method `st()`, which means "subject to". This method allows users to define their constraints in different ways.


```python
model.st(A @ x <= c)                    # define one constraint

model.st(y.sum() >= 1,
         y.sum(axis=1) >= 0,
         A*y + x >= 0)                  # define multiple constraints

model.st(x[i] <= i for i in range(3))   # define constraints by a loop
```

### Convex Expressions and Convex Constraints

The RSOME package also supports several functions for specifying convex expressions and constraints. The definition and syntax of these functions are also similar to the NumPy package, see the tables below.

|Convex Function| Output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Remarks  
|:-------|:--------------|:-----------------|
|`abs(x)`|The element-wise absolute values of `x`. | |
|`entropy(x)`|The entropic expression `-sum(x * log(x))`. |`x` must be a vector.|
|`exp(x)`|The element-wise natural exponential of `x`. | |
|`log(x)`|The element-wise natural logarithm of `x`.| |
|`norm(x, degree)`| The norm of the `x`. |`x` must be a vector. `degree` can be 1, 2, or `numpy.inf`. The default value is `degree=2`, indicating an Euclidean norm. |
|`pexp(x, y)`|The element-wise perspective natural exponential `y * exp(x/y)`. | |
|`plog(x, y)`|The element-wise perspective natural logarithm `y * log(x/y)`. | |
|`quad(x, Q)`|The quadratic term `x @ Q @ x`. |`x` must be a 1-D array, and `Q` must be a positive semidefinite matrix (2-D array).|
|`square(x)`|The element-wise squared values of `x`. | |
|`sumsqr(x1, x2, ...)`|The sum of squares of elements in arrays `x1`, `x2`, ... | |


|Convex Constraints| Output| Remarks  
|:-------|:--------------|:-----------------|
|`expcone(x, y, z)`| The exponential cone constraint `z * exp(x/z) <= y`. |`x` and `z` must be scalars.|
|`kldiv(p, q, r)`| The KL divergence constraint `sum(p*log(p/q)) <= r`. |`p` and `q` are vectors,<br>and `r` is a scalar.|
|`rsocone(x, y, z)`| The rotated conic constraint `sumsqr(x) <= y*z`. |`x` must be a vector.|


Examples of specifying convex constraints are provided below.


```python
import rsome as rso
from rsome import ro
from numpy import inf

model = ro.Model()
x = model.dvar(I)                       # define x as a 1D array of I variables
y = model.dvar((J, I))                  # define y as a 2D array of JxI variables
z = model.dvar()                        # define z as a scalar variable

model.st(abs(x) <= 2)                   # constraints with absolute terms
model.st(rso.sumsqr(x) <= 10)           # a constraint with sum of squares
model.st(rso.square(y) <= 5)            # constraints with squared terms
model.st(rso.norm(y[:, 0]) <= 1)        # a constraint with 2-norm terms
model.st(rso.norm(x, 1) <= y[0, 0])     # a constraint with 1-norm terms
model.st(rso.norm(x, inf) <= x[0])      # a constraint with infinity norm
model.st(rso.quad(x, Q) + x[1] <= x[0]) # a constraint with a quadratic term
model.st(rso.rsocone(x, y[0, 0], z))    # a constraint with a rotated second-order cone
model.st(rso.expcone(x, x[0], 1.5))     # an exponential cone constraint
model.st(rso.exp(x) <= 3.5)             # constraints with exponential terms
model.st(rso.log(x) >= 1.2)             # constraints with logarithm terms
model.st(rso.entropy(x) >= x[1])        # constraints with entropy expressions
model.st(rso.kldiv(x, 1/I, 0.01))       # a KL divergence constraint
```

Note that all functions above can only be used in constructing convex constraints, so convex functions cannot be applied in equality constraints, and they cannot be used for concave inequalities, such as `abs(x) >= 2` is invalid and gives an error message.


### Matrix Operation Functions

In dealing with matrices (2-D arrays), RSOME supports basic operations like calculating the trace, retrieving the diagonal and upper/lower triangular elements, and concatenating arrays. 

|Matrix Operations&nbsp;| Output| Remarks  
|:-------|:--------------|:-----------------|
|`concat(arrays, axis)`| Concatenated arrays | `arrays` is a collection of arrays to be conca-<br>tenated, and `axis` specifies the axis along<br>which the arrays will be joined. |
|`cstack(c1, ..., cn)`|An array formed by stacking the<br>given arrays along axis 1. |If a given argument `c1, ..., cn` is a list of<br>arrays, they will be stacked first along axis 0. |
|`diag(x, k, fill)`| The diagonal elements of a 2-D<br>array `x`. |The integer `k` (`k=0` by default) specifies the<br>shifts of the taken diagonal elements. The<br>boolean value `fill` (`fill=False` by default)<br>specifies if the non-diagonal elements are<br>filled with zeros.|
|`rstack(r1, ..., rn)`|An array formed by stacking the<br>given arrays along axis 0. |If a given argument `r1, ..., rn` is a list of<br>arrays, they will be stacked first along axis 1. |
|`trace(x)`| The trace of a 2-D array `x`. ||
|`tril(x, k)`| The lower triangular elements of a<br>2-D array `x`. |The integer `k` (`k=0` by default) specifies the<br>shifts of the taken triangular elements.|
|`triu(x, k)`| The upper triangular elements of a<br>2-D array `x`. |The integer `k` (`k=0` by default) specifies the<br>shifts of the taken triangular elements.|

RSOME also supports defining semidefiniteness constraints that enforce an array `X` to be positive semidefinite (`X >> 0`) or negative semidefinite (`X << 0`). For instance, the constraint

$$
\left(
\begin{array}{cc}
A & Z \\
Z^{\top} & \text{diag}(Z)
\end{array}
\right) \succeq 0
$$

with \\(A\in\mathbb{R}^{n\times n}\\) and \\(Z\\) being a lower triangular matrix, can be written as the following Python code,

```python
import rsome as rso
from rsome import ro

model = ro.Model()

A = model.dvar((n, n))
Z = rso.tril(model.dvar((n, n)))

model.st(rso.rstack([A, Z], 
                    [Z.T, rso.diag(Z, fill=True)]) >> 0)
```

or equivalently,

```python
import rsome as rso
from rsome import ro

model = ro.Model()

A = model.dvar((n, n))
Z = rso.tril(model.dvar((n, n)))

model.st(rso.cstack([A, Z.T], 
                    [Z, rso.diag(Z, fill=True)]) >> 0)
```

In this example, the `T` attribute of an RSOME array is consistent with the transpose operations of NumPy arrays. Functions `rstack()` and `cstack()` are used to create new arrays by stacking rows (along axis 0) or columns (along axis 1) together, respectively. 



## Standard Forms and Solutions <a name="section1.3"></a>

All RSOME models are transformed into their standard forms, which are then solved via the solver interface. The standard form can be retrieved by the `do_math()` method of the model object.

```
do_math(self, primal=True)
    Return the linear, or conic programming problem as the standard
    formula or deterministic counterpart of the model.
    
    Parameters
    ----------
    primal : bool, default True
        Specify whether return the primal formula of the model.
        If primal=False, the method returns the dual formula.
    
    Returns
    -------
    prog : GCProg
        A conic programming problem.
```

You may use the `do_math()` method together with the `show()` method to display important information on the standard form, <i>i.e.</i>, the coefficients of the objective function and constraints, variable bounds and types.


```python
from rsome import ro
import rsome as rso
import numpy as np


n = 3
np.random.seed(1)
c = np.random.normal(size=n)

model = ro.Model()
x = model.dvar(n)

model.max(c @ x)
model.st(rso.norm(x) <= 1)

primal = model.do_math()            # standard form of the primal problem
dual = model.do_math(primal=False)  # standard form of the dual problem
```

The variables `primal` and `dual` represent the standard forms of the primal and dual problems, respectively.

```python
primal
```

    Conic program object:
    =============================================
    Number of variables:           8
    Continuous/binaries/integers:  8/0/0
    ---------------------------------------------
    Number of linear constraints:  5
    Inequalities/equalities:       2/3
    Number of coefficients:        11
    ---------------------------------------------
    Number of SOC constraints:     1
    ---------------------------------------------
    Number of ExpCone constraints: 0
    ---------------------------------------------
    Number of PSCone constraints:  0


```python
dual
```


    Conic program object:
    =============================================
    Number of variables:           5
    Continuous/binaries/integers:  5/0/0
    ---------------------------------------------
    Number of linear constraints:  4
    Inequalities/equalities:       0/4
    Number of coefficients:        7
    ---------------------------------------------
    Number of SOC constraints:     1
    ---------------------------------------------
    Number of ExpCone constraints: 0
    ---------------------------------------------
    Number of PSCone constraints:  0


More details on the standard forms can be retrieved by the method `show()`, and the problem information is summarized in a `pandas.DataFrame` data table.


```python
primal.show()
```




<div>
<table border="1" class="dataframe mystyle">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>sense</th>
      <th>constant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Obj</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>LC1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>==</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>LC2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>==</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>LC3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>==</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>LC4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC5</th>
      <td>-1</td>
      <td>-1.624345</td>
      <td>0.611756</td>
      <td>0.528172</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>&lt;=</td>
      <td>0</td>
    </tr>
    <tr>
      <th>QC1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>&lt;=</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UB</th>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>LB</th>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Type</th>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>




```python
dual.show()
```




<div>
<table border="1" class="dataframe mystyle">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>sense</th>
      <th>constant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Obj</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>LC1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.624345</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.611756</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.528172</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>QC1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>&lt;=</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UB</th>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>LB</th>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Type</th>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>

Besides being returned as a `pandas.DataFrame` data table, the standard form can also be saved as a `.lp` file using the `to_lp()` method.

```
to_lp(name='out')
Export the standard form of the optimization model as a .lp file.

    Parameters
    ----------
    name : file name of the .lp file

    Notes
    -----
    There is no need to specify the .lp extension. The default file name
    is "out".
```

The code segment below exports the standard form of the model as a file named "model.lp".

```python
model.do_math().to_lp('model')
```

The standard form of a model can be solved via calling the `solve()` method of the model object. Arguments of the `solve()` method are listed below.

```
solve(self, solver=None, display=True, log=False, params={})
    Solve the model with the selected solver interface.
    
    Parameters
    ----------
    solver : {None, lpg_solver, clp_solver, ort_solver, eco_solver
              cpx_solver, grb_solver, msk_solver, cpt_solver}
        Solver interface used for model solution. Use the default
        solver if solver=None.
    display : bool
        True for displaying the solution information. False for hiding
        the solution information.
    log : bool
        True for printing the log information. False for hiding the log
        information. So far the argument only applies to Gurobi, CPLEX,
        and Mosek.
    params : dict
        A dictionary that specifies parameters of the selected solver.
        So far the argument only applies to Gurobi, CPLEX, and Mosek.
```

The `solve()` method calls for external solvers to solve the optimization problem. The first argument `solver` is used to specify the selected solver interface. If the solver is unspecified, the default solver imported from the the `scipy.optimize` is used to solve the RSOME model. Other open-source and commercial solvers can also be used in RSOME. Details of the interfaces for calling these external solvers are presented in the table below.  

| Solver | License  type | Required version | RSOME interface | Second-order cone constraints| Exponential cone constraints | Semidefiniteness constraints
|:-------|:--------------|:-----------------|:----------------|:------------------------|:---------------------|:--------------|
|[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)| Open-source | >= 1.9.0 | `lpg_solver` | No | No | No |
|[CyLP](https://github.com/coin-or/cylp)| Open-source | >= 0.9.0 | `clp_solver` | No | No | No |
|[OR-Tools](https://developers.google.com/optimization/install) | Open-source | >= 7.5.7466 | `ort_solver` | No | No | No |
|[ECOS](https://github.com/embotech/ecos-python) | Open-source | >= 2.0.10 | `eco_solver` | Yes | Yes | No |
|[Gurobi](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html)| Commercial | >= 9.1.0 | `grb_solver` | Yes | No | No |
|[Mosek](https://docs.mosek.com/9.2/pythonapi/install-interface.html) | Commercial | >= 10.0.44 | `msk_solver` | Yes | Yes | Yes |
|[CPLEX](https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) | Commercial | >= 12.9.0.0 | `cpx_solver` | Yes | No | No |
|[COPT](https://www.shanshu.ai/copt) | Commercial | >= 6.5.3 | `cpt_solver` | Yes | No | No |


The model above involves second-order cone constraints, so we could use ECOS, Gurobi, Mosek, CPLEX, or COPT to solve it. The interfaces for these solvers are imported by the following commands.

```python
from rsome import eco_solver as eco
from rsome import grb_solver as grb
from rsome import msk_solver as msk
from rsome import cpx_solver as cpx
from rsome import cpt_solver as cpt
```

The interfaces can be then used to attain the solution.

```python
model.solve(eco)
```    
    Being solved by ECOS...
    Solution status: Optimal solution found
    Running time: 0.0003s


```python
model.solve(grb)
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0009s


```python
model.solve(msk)
```

    Being solved by Mosek...
    Solution status: Optimal
    Running time: 0.0230s


```python
model.solve(cpx)
```

    Being solved by CPLEX...
    Solution status: optimal
    Running time: 0.0135s


```python
model.solve(cpt)
```

    Cardinal Optimizer v6.5.3. Build date Apr 28 2023
    Copyright Cardinal Operations 2023. All Rights Reserved

    Being solved by COPT...
    Solution status: 1
    Running time: 0.0028s

It can be seen that as the model is solved, a three-line message is displayed in terms of 1) the solver used for solving the model; 2) the solution status; and 3) the solution time. This three-line message can be disabled by specifying the second argument `display` to be `False`.

The third argument `params` is used to tune solver parameters. The current RSOME package enables users to adjust parameters for Gurobi, MOSEK, and CPLEX. The `params` argument is a `dict` type object in the format of `{<param1>: <value1>, <param2>: <value2>, <param3>: <value3>, ..., <paramk>: <valuek>}`. Information on solver parameters and their valid values are provided below. Please make sure that you are specifying parameters with the correct data type, otherwise error messages might be raised.
- Gurobi parameters: [https://www.gurobi.com/documentation/9.1/refman/parameters.html](https://www.gurobi.com/documentation/9.1/refman/parameters.html)
- MOSEK parameters: [https://docs.mosek.com/latest/pythonapi/parameters.html](https://docs.mosek.com/latest/pythonfusion/parameters.html)
- CPLEX parameters: [https://www.ibm.com/docs/en/icos/12.7.1.0?topic=cplex-list-parameters](https://www.ibm.com/docs/en/icos/12.7.1.0?topic=cplex-list-parameters)


For example, the following code solves the problem using Gurobi, MOSEK, and CPLEX, respectively, with the relative MIP gap tolerance to be `1e-3`.

```python
model.solve(grb, params={'MIPGap': 1e-3})
model.solve(msk, params={'mioTolRelGap': 1e-3})
model.solve(cpx, params={'mip.tolerances.mipgap': 1e-3})
```

Once the optimization problem is solved, you may use the command `model.get()` to retrieve the optimal objective value. The optimal solution of the variable `x` can be attained as an array by calling `x.get()`. The `get()` method raises an error message if the optimization problem is unsolved, or the optimal solution cannot be found due to infeasibility, unboundedness, or numerical issues. 


## Application Examples <a name="section1.4"></a>

### [Mean-Variance Portfolio](example_mv_portfolio)
### [Integer Programming for Sudoku](example_sudoku)
### [Optimal DC Power Flow](example_opf)
### [The Unit Commitment Problem](example_ucp)
### [Box with the Maximum Volume](example_max_volume_box)
### [Minimal Enclosing Ellipsoid](example_min_ellipsoid)

## Reference

<a id="ref1"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Getting Started

RSOME is an open-source algebraic library for generic optimization modeling. It is aimed at providing a fast and highly readable modeling environment for the state-of-the-art robust stochastic optimization framework.

This guide introduces the main components, basic data structures, and syntax rules of the RSOME package. As for the package and solver installation, please refer to our [home page](index) for more information.

## Modeling environments <a name="section1.1"></a>

The current version of RSOME provides several layers of modeling environments, as illustrated by the structure diagram below.

<img src="rsome_modules.png" width=600/>

The structure diagram also shows that `lp` for linear programs is a bottom layer modeling environment. Problems can be solved by the `lp` layer are a subset of the `socp` layer. Similarly, problems can be solved by the `socp` layer are a subset of the higher level `ro` environment. The top layer `dro` for distributionally robust optimization is the most general modeling environment, with all lower layers to be special cases of this general framework.

The syntax rules of `lp`, `socp`, and `ro` are very similar, so we would focus on `ro` as a more general modeling environment for all robust and deterministic optimization problems. The `dro` is specially designed for distributionally robust optimization problems, with the worst-case expectations to be considered in the objective function or constraints. It will be introduced separately.

## Introduction to the `rsome.ro` environment <a name="section1.2"></a>

### Models

In RSOME, all optimization models are specified based on a <code>Model</code> type object. Such an object is created by the constructor <code>Model()</code> imported from the <code>rsome.ro</code> modeling environment.


```python
from rsome import ro            # Import the ro modeling tool

model = ro.Model('My model')    # Create a Model object
```

The code above defines a new <code>Model</code> object <code>model</code>, with the name specified to be <code>'My model'</code>. You could also leave the name unspecified and the default name is <code>None</code>.

### Decision variables

Decision variables of a model can be defined by the method <code>dvar()</code>.
```
dvar(shape=(1,), vtype='C', name=None, aux=False) method of rsome.ro.Model instance
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
```


Similar to the <code>numpy.ndarray</code> data objects, variables in RSOME can be formatted as N-dimensional arrays, and the dimensional number is determined by the length of the tuple type attribute <code>shape</code>. Some important attributes of RSOME variables are provided below. It can be seen that they are consistent with the <code>numpy.ndarray</code> class.

```
Affine
    An Affine type object as the transpose of the variable array
ndim : int
    Number of variable array dimensions
shape : int or tuple
    The shape of the variable array
size : int
    Number of variables in the array
vtype : {'C', 'B', 'I'}
    Type of variables
name : str
    Name of the variable array
```

A few examples of decision variables are presented below.


```python
x = model.dvar(3, vtype='I')    # Integer variables as a 1D array
y = model.dvar((3, 5), 'B')     # Binary variables as a 2D array
z = model.dvar((2, 3, 4, 5))    # Continuous variables as a 4D array
```

### Affine operations and linear constraints

Any affine operations of a variable array would create an <code>Affine</code> type object, as shown by the sample code below.


```python
model = ro.Model()

x = model.dvar(3)
y = model.dvar((3, 5))
z = model.dvar((2, 3, 4))

type(3*x + 1)               # Display the Affine type
```



    rsome.lp.Affine



The <code>Affine</code> objects are also compatible with the standard NumPy array syntax, like reshaping, element-wise operations, matrix calculation rules, broadcasting, indexing and slicing. A few examples are provided below.


```python
import numpy as np

a = np.ones(3)
expr1 = a @ x                       # Matrix multiplication
print(expr1)

b = np.arange(15).reshape((3, 5))
expr2 = b + y                       # Element-wise operation
print(expr2)

c = np.arange(12).reshape((3, 4))
expr3 = c * z                       # Broadcasting
print(expr3)

expr4 = x.reshape((3, 1)) + y       # Reshape and broadcasting
print(expr4)

expr5 = x + y[:, 2].T               # Slicing and transpose
print(expr5)
```

    1 affine expressions
    3x5 affine expressions
    2x3x4 affine expressions
    3x5 affine expressions
    3 affine expressions


These affine expressions can be then used in specifying the objective function of the optimization model. Please note that the objective function must be one affine expression. In other words, the <code>size</code> attribute of the expression must be one, otherwise an error message would be generated.


```python
model.min(a @ x)        # Minimize the objective function a @ x
model.max(a @ x)        # Maximize the objective function a @ x
```

Model constraints can be specified by the method <code>st()</code>, which means "subject to". This method allows users to define their constraints in different ways.


```python
model.st(c * z <= 0)        # Define one constraint

model.st(x >= 0,
         x <= 10,
         y >= 0,
         y <= 20)           # Define multiple constraints as a tuple

model.st(x[i] <= i
         for i in range(3)) # Define constraints via comprehension
```

### Convex functions and convex constraints

The RSOME package also supports several convex functions for specifying convex constraints. The definition and syntax of these functions are also consistent with the NumPy package.

- **<code>abs()</code> for absolute values**: the function <code>abs()</code> returns the element-wise absolute value of an array of variables or affine expressions.

- **<code>square()</code> for squared values**: the function <code>square</code> returns the element-wise squared values of an array of variables or affine expressions.

- **<code>sumsqr()</code> for sum of squares**: the function <code>sumsqr()</code> returns the sum of squares of a vector, which is a one-dimensional array, or an array with its <code>size</code> to be the same as maximum <code>shape</code> value.

- **<code>norm()</code> for norms of vectors**: the function <code>sumsqr()</code> returns the first, second, or infinity norm of a vector. Users may use the second argument <code>degree</code> to specify the degree of the norm function. The default value of the <code>degree</code> argument is 2. Examples of specifying convex constraints are provided below.


```python
import rsome as rso
from numpy import inf

model.st(abs(z) <= 2)               # Constraints with absolute terms
model.st(rso.sumsqr(x) <= 10)       # A Constraint with sum of squares
model.st(rso.square(y) <= 5)        # Constraints with squared terms
model.st(rso.norm(z[:, 2, 0]) <= 1) # A Constraint with 2-norm terms
model.st(rso.norm(x, 1) <= y[0, 0]) # A Constraint with 1-norm terms
model.st(rso.norm(x, inf) <= x[0])  # A Constraint with infinity norm
```

Please note that all functions above can only be used in convex functions, so convex function cannot be applied in equality constraints, and these functions cannot be used for concave inequalities, such as <code>abs(x) >= 2</code> is invalid and gives an error message.

## Standard forms and solutions <a name="section1.3"></a>

All RSOME models are transformed into their standard forms, which are then solved via the solver interface. The standard form can be retrieved by the <code>do_math()</code> method of the model object.

```
Model.do_math(primal=True)
    Returns a SOCProg type object representing the standard form
    as a second-order cone program. The parameter primal controls
    the returned formula is for the primal or the dual problem.
```

You may use the <code>do_math()</code> method together with the <code>show()</code> method to display important information on the standard form, i.e., the objective function, linear and second-order cone constraints, bounds and variable types.


```python
import rsome as rso
import numpy.random as rd
from rsome import ro

n = 3
c = rd.normal(size=n)

model = ro.Model()
x = model.dvar(n)

model.max(c @ x)
model.st(rso.norm(x) <= 1)

primal = model.do_math()            # Standard form of the primal problem
dual = model.do_math(primal=False)  # Standard form of the dual problem
```

The variables `primal` and `dual` represent the standard forms of the primal and dual problems, respectively.

```python
primal
```




    Second order cone program object:
    =============================================
    Number of variables:          8
    Continuous/binaries/integers: 8/0/0
    ---------------------------------------------
    Number of linear constraints: 5
    Inequalities/equalities:      1/4
    Number of coefficients:       11
    ---------------------------------------------
    Number of SOC constraints:    1


```python
dual
```


    Second order cone program object:
    =============================================
    Number of variables:          5
    Continuous/binaries/integers: 5/0/0
    ---------------------------------------------
    Number of linear constraints: 4
    Inequalities/equalities:      0/4
    Number of coefficients:       7
    ---------------------------------------------
    Number of SOC constraints:    1


More details on the standard forms can be retrieved by the method `show()`, and the problem information is summarized in a `pandas.DataFrame` table.


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
      <td>0.585058</td>
      <td>0.0693541</td>
      <td>-0.7489</td>
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
      <td>0.585058</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0693541</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.7489</td>
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

The standard form of the model can be solved via calling the `solve()` method of the model object. Arguments of the `solve()` method are listed below.

    solve(solver, display=True, export=False) method of rsome.ro.Model instance
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



It can be seen that the user needs to specify the `solver` argument for selecting the solver interface when calling the `solve()` method. The current version of RSOME provides solver interfaces for Gurobi and MOSEK.

```python
from rsome import grb_solver as grb
from rsome import msk_solver as msk
```

The interfaces can be used to attain the solution.

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
    Solution status: optimal
    Running time: 0.0210s



The other two arguments control the display and export options of the solution. Once the solution completes, you may use the command `model.get()` to retrieve the optimal objective value. The optimal solution of the variable `x` can be attained as an array by calling `x.get()`. No optimal value or solution can be retrieved if the problem is infeasible, unbounded, or terminated by a numeric issue.  

## Application examples <a name="section1.4"></a>

### [Mean-variance portfolio optimization](example_mv_portfolio)
### [Integer programming for Sudoku](example_sudoku)

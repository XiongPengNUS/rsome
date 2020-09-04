<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Quickstart Tutorial

## Models

In RSOME, all optimization models are specified based on a <code>Model</code> type object. Such an object is created by the constructor <code>Model()</code> imported from the <code>rsome.ro</code> toolbox.


```python
from rsome import ro            # Import the ro modeling tool

model = ro.Model('My model')    # Create a Model object
```

The code above defines a new <code>Model</code> object <code>model</code>, with the name specified to be <code>'My model'</code>. You could also leave the name unspecified and the default name is <code>None</code>.

## Decision variables

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

## Affine operations and linear constraints

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

## Convex functions and convex constraints

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

## Standard form and solutions

As mentioned in the previous sections, an optimization model is transformed into a standard form, which is then solved via the solver interface. The standard form can be retrieved by the <code>do_math()</code> method of the model object.

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



The other two arguments control the display and export options of the solution. Since there is no significant difference in calling various solvers, we would constantly use the Gurobi solver in the remaining part of the manual for demonstration.


## Application examples

### Mean-variance portfolio optimization

In this example, we consider a portfolio construction problem discussed in [Robust Solutions of Uncertain Linear Programs](https://www2.isye.gatech.edu/~nemirovs/stablpn.pdf). Suppose there are \\(n=150\\) stocks, and each stock \\(i\\) has the mean return to be \\(p_i\\) and the standard deviation to be \\(\sigma_i\\). Let \\(x_i\\) be the fraction of wealth invested in stock \\(i\\), a classic approach is to formulate the problem as a quadratic program, where a mean-variance objective function is maximized:

$$
\begin{align}
\max~&\sum\limits_{i=1}^np_ix_i - \phi \sum\limits_{i=1}^n \sigma_i^2x_i^2 \\
\text{s.t.}~&\sum\limits_{i=1}^nx_i = 1 \\
& x_i \geq 1, ~\forall i = 1, 2, ..., n,
\end{align}
$$

with the constant \\(\phi=5\\), and the means and standard deviations are specified to be

$$
\begin{align}
&p_i = 1.15 + i\frac{0.05}{150} \\
&\sigma_i = \frac{0.05}{450}\sqrt{2in(n+1)}.
\end{align}
$$

The quadratic program can be implemented by the following code segment.


```python
import rsome as rso
import numpy as np
from rsome import ro
from rsome import grb_solver as grb

n = 150                                     # Number of stocks
i = np.arange(1, n+1)                       # Indices of stocks
p = 1.15 + i*0.05/150                       # Mean returns
sigma = 0.05/450 * (2*i*n*(n+1))**0.5       # Standard deviations of returns
phi = 5                                     # Constant phi

model = ro.Model('mv-portfolio')

x = model.dvar(n)                           # Fractions of investment

model.max(p@x - phi*rso.sumsqr(sigma*x))    # Mean-variance objective
model.st(x.sum() == 1)                      # Summation of x is one
model.st(x >= 0)                            # x is non-negative

model.solve(grb)
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0028s


The optimal investment decision and the mean-variance objective value are shown below.


```python
import matplotlib.pyplot as plt

obj_val = model.get()               # The optimal objective value
x_sol = x.get()                     # The optimal investment decision

plt.plot(range(1, n+1), x_sol,
         linewidth=2, color='b')
plt.xlabel('Stocks')
plt.ylabel('Fraction of investment')
plt.show()
print('Objective value: {0:0.4f}'.format(obj_val))
```


![](example_socp.png)


    Objective value: 1.1853


### Integer programming for Sudoku

In this section we will use a [Sudoku](https://en.wikipedia.org/wiki/Sudoku) game to illustrate how to use integer and multi-dimensional arrays in RSOME. Sudoku is a popular number puzzle. The goal is to place the digits in \[1,9\] on a nine-by-nine grid, with some of the digits already filled in. Your solution must satisfy the following four rules:

1. Each cell contains an integer in \[1,9\].
2. Each row must contain each of the integers in \[1,9\].
3. Each column must contain each of the integers in \[1,9\].
4. Each of the nine 3x3 squares with bold outlines must contain each of the integers in \[1,9\].

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg/1280px-Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg.png" width=200>
</p>

The Sudoku game can be considered as a optimization with the objective to be zero and constraints used to fulfill above rules. Consider a binary variable \\(x_{ijk}\in \{0, 1\}\\), where \\(\forall i \in [0, 8]\\), \\(j \in [0, 8]\\), \\(k \in [0, 8]\\). It equals to one if an integer \\(k+1\\) is placed in a cell at the \\(i\\)th row and \\(j\\)th column. Let \\(a_{ij}\\) be the known number at the \\(i\\)th row and \\(j\\)th column, where \\(i\in\mathcal{I}\\) and \\(j\in\mathcal{J}\\), the Sudoku game can be written as the following integer programming problem

$$
\begin{align}
\min~&0 \\
\text{s.t.}~& \sum\limits_{i=0}^8x_{ijk} = 1, \forall j \in [0, 8], k \in [0, 8] \\
& \sum\limits_{j=0}^8x_{ijk} = 1, \forall i \in [0, 8], k \in [0, 8] \\
& \sum\limits_{k=0}^8x_{ijk} = 1, \forall i \in [0, 8], j \in [0, 8] \\
& x_{ij(a_{ij}-1)} = 1, \forall i \in \mathcal{I}, j \in \mathcal{J} \\
& \sum\limits_{m=0}^2\sum\limits_{n=0}^2x_{(i+m), (j+m), k} = 1, \forall i \in \{0, 3, 6\}, j \in \{0, 3, 6\}, k \in [0, 8]
\end{align}
$$

In the following code, we are using RSOME to implement such a model.


```python
import rsome as rso
import numpy as np
from rsome import ro
from rsome import grb_solver as grb

# A Sudoku puzzle
# Zeros represent unknown numbers
puzzle = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 2],
                   [6, 0, 0, 1, 9, 5, 0, 0, 0],
                   [0, 9, 8, 0, 0, 0, 0, 6, 0],
                   [8, 0, 0, 0, 6, 0, 0, 0, 3],
                   [4, 0, 0, 8, 0, 3, 0, 0, 1],
                   [7, 0, 0, 0, 2, 0, 0, 0, 6],
                   [0, 6, 0, 0, 0, 0, 2, 8, 0],
                   [0, 0, 0, 4, 1, 9, 0, 0, 5],
                   [0, 0, 0, 0, 8, 0, 0, 7, 9]])

# Create model and binary decision variables
model = ro.Model()
x = model.dvar((9, 9, 9), vtype='B')

# Objective is set to be zero
model.min(0 * x.sum())

# Constraints 1 to 3
model.st(x.sum(axis=0) == 1,
         x.sum(axis=1) == 1,
         x.sum(axis=2) == 1)

# Constraints 4
i, j = np.where(puzzle)
model.st(x[i, j, puzzle[i, j]-1] == 1)

# Constraints 5
for i in range(0, 9, 3):
    for j in range(0, 9, 3):
        model.st(x[i: i+3, j: j+3, :].sum(axis=(0, 1)) == 1)

# Solve the integer programming problem
model.solve(grb)
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0017s


The binary variable \\(x_{ijk}\\) is defined to be a three-dimensional array <code>x</code> with the shape to be <code>(9, 9, 9)</code>. Please note that in RSOME, the objective function cannot be specified as a numeric constant, we then use the expression <code>0 * x.sum()</code> as the objective. Based on the decision variable <code>x</code>, each set of constraints can be formulated as the array form by using the <code>sum</code> method. The method <code>sum()</code> in RSOME is consistent with that in NumPy, where you may use the <code>axis</code> argument to specify along which axis the sum is performed.

The Sudoku problem and the its solution are presented below.


```python
print(puzzle)                   # Display the Sudoku puzzle
```

    [[5 3 0 0 7 0 0 0 2]
     [6 0 0 1 9 5 0 0 0]
     [0 9 8 0 0 0 0 6 0]
     [8 0 0 0 6 0 0 0 3]
     [4 0 0 8 0 3 0 0 1]
     [7 0 0 0 2 0 0 0 6]
     [0 6 0 0 0 0 2 8 0]
     [0 0 0 4 1 9 0 0 5]
     [0 0 0 0 8 0 0 7 9]]



```python
x_sol = x.get().astype('int')   # Retrieve the solution as integers

print((x_sol * np.arange(1, 10).reshape((1, 1, 9))).sum(axis=2))
```

    [[5 3 4 6 7 8 9 1 2]
     [6 7 2 1 9 5 3 4 8]
     [1 9 8 3 4 2 5 6 7]
     [8 5 9 7 6 1 4 2 3]
     [4 2 6 8 5 3 7 9 1]
     [7 1 3 9 2 4 8 5 6]
     [9 6 1 5 3 7 2 8 4]
     [2 8 7 4 1 9 6 3 5]
     [3 4 5 2 8 6 1 7 9]]

Please note that in defining "Constraints 4", variables `i` and `j` represent the row and column indices of the fixed elements, which can be retrieved by the `np.where()` function. An alternative approach is to use the boolean indexing of arrays, as the code below.

```python
# An alternative approach for Constraints 4
is_fixed = puzzle > 0
model.st(x[is_fixed, puzzle[is_fixed]-1] == 1)
```

The variable `is_fixed` an array with elements to be `True` if the number is fixed and `False` if the number is unknown. Such a boolean type array can also be used as indices, thus defining the same constraints.

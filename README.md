<img src="https://github.com/XiongPengNUS/test/blob/master/rologo.jpeg?raw=true![image.png](attachment:image.png)" width=75 align="left">

# ROAD: Robust Optimization for Array Data

ROAD (Robust Optimization for Array Data) is a open-source Python package for operations research and generic optimization modeling. ROAD models are constructed by variables, expressions, and constraints formatted as N-dimensional arrays, which are consistent with the NumPy library in syntax and operations, such as indexing and slicing, element-wise operations, broadcasting, and matrix calculation rules. It thus provides a convenient and highly readable way in developing optimization models and applications. 

The current version of ROAD supports deterministic linear/second-order cone programs and robust optimization problems. An interface with Gurobi solver is also integrated for the solution of optimization models. Distributionally robust optimization modeling tools based on the [robust stochastic optimization (RSO) framework](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603?af=R) is now under development. Other solver interfaces will be included in the future.

## Introduction

### Installing ROAD and solvers

The ROAD package can be installed with the <code>pip</code> command: 
***
**`pip install road`**
***

For the current version, the Gurobi solve is also needed for solving the optimization model, and you may follow [these steps](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html) to complete the solver installation. 

### The Dao of ROAD

The ROAD package is largely inspired by [ROME](https://robustopt.com/), the very first software toolbox for robust optimization. We also learned many hard lessons in developing the MATLAB package [RSOME](https://www.rsomerso.com/), hence the "Dao of ROAD", which can be imported from the ROAD package. 


```python
from road import dao
```

    The DAO of ROAD: 
    ROME was not built in one day.
    All ROADs lead to ROME.
    Matlab is RSOME!
    The ROAD in Python is more than RSOME!
    
    
    ROME: https://robustopt.com/
    RSOME: https://www.rsomerso.com
    ROAD: https://github.com/XiongPengNUS/road


### Getting started

The ROAD package supports specifying models using highly readable algebraic expressions that are consistent with NumPy syntax. A very simple linear program example is provided below,



{
 "metadata": {
  "name": "Typesetting Math Using MathJax"
 },
 "nbformat": 3,
 "nbformat_minor": 0,
 "worksheets": [
  {
   "cells": [
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "The Markdown parser included in IPython is MathJax-aware.  This means that you can freely mix in mathematical expressions using the [MathJax subset of Tex and LaTeX](http://docs.mathjax.org/en/latest/tex.html#tex-support).  [Some examples from the MathJax site](http://www.mathjax.org/demos/tex-samples/) are reproduced below, as well as the Markdown+TeX source."
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "# Motivating Examples\n",
      "\n",
      "---\n",
      "\n",
      "## The Lorenz Equations\n",
      "### Source\n",
      "```\\begin{aligned}\n",
      "\\dot{x} & = \\sigma(y-x) \\\\\n",
      "\\dot{y} & = \\rho x - y - xz \\\\\n",
      "\\dot{z} & = -\\beta z + xy\n",
      "\\end{aligned}\n",
      "```\n",
      "### Display\n",
      "\\begin{aligned}\n",
      "\\dot{x} & = \\sigma(y-x) \\\\\n",
      "\\dot{y} & = \\rho x - y - xz \\\\\n",
      "\\dot{z} & = -\\beta z + xy\n",
      "\\end{aligned}"
     ]
    },
   ],
   "metadata": {}
  }
 ]
}



{
 "worksheets": [
  {
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Maxwell's Equations\n",
      "### Source\n",
      "```\\begin{align}\n",
      "\max ~&3x + 4y \\ \n",
      "\text{s.t.}~&2.5x + y \leq 20 \\ \n", 
      "&5x + 3y \leq 30 \\ \n", 
      "&x + 2y \leq 16 \\ \n", 
      "&|y| \leq 2, \n", 
      "\\end{align}"
     ]
    },
    }]
}
    

and it is used to illustrate the steps of solving optimization models. 


```python
from road import ro                 # Import the ro modeling tool 
from road import grb_solver as grb  # Import Gurobi solver interface

model = ro.Model('LP model')        # Create a Model object
x = model.dvar()                    # Define a decision variable x
y = model.dvar()                    # Define a decision variable y

model.max(3*x + 4*y)                # Maximize the objective function
model.st(2.5*x + y <= 20)           # Specify the 1st constriants
model.st(5*x + 3*y <= 30)           # Specify the 2nd constraints
model.st(x + 2*y <= 16)             # Specify the 3rd constraints
model.st(abs(y) <= 2)               # Specify the 4th constraints

model.solve(grb)                    # Solve the model with Gurobi
```

    Using license file /Users/pengxiong/gurobi.lic
    Academic license - for non-commercial use only
    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0005s


In this sample code, a model object is created by calling the constructor <code>Model()</code> imported from the <code>road.ro</code> toolbox. Based on the model object, decision variables <code>x</code> and <code>y</code> are created by the method <code>dvar()</code>. These variables are then used in specifying the objective function and model constraints. The last step is calling the <code>solve()</code> method to solve the problem via the imported solver interface <code>grb</code>. Once the solution completes, a message showing the solution status and running time will be printed.

According to the [Gurobi solution status](https://www.gurobi.com/documentation/9.0/refman/optimization_status_codes.html), the status code <code>2</code> suggests that the problem was solved to optimality (subject to tolerances), and an optimal solution is available. The optimal solution and the corresponding objective value can be attained by the <code>get()</code> method. 


```python
print('x:', x.get())
print('y:', y.get())
print('Objective:', model.get())
```

    x: [4.8]
    y: [2.]
    Objective: 22.4


The example above shows how to specify an optimization model with highly readable algebraic expressions. The package is capable of transforming the specified model into a standard formula, which can be recognized and solved by the solver. 

Users could retrieve the standard formula by calling the method <code>do_math()</code> of the model object. 


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
    


We also provide a debugging method <code>show()</code> to display the information of the standard formula as a data frame. 


```python
formula.show()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>sense</th>
      <th>constants</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Obj</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>LC1</th>
      <td>0</td>
      <td>2.5</td>
      <td>1</td>
      <td>&lt;=</td>
      <td>20</td>
    </tr>
    <tr>
      <th>LC2</th>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>&lt;=</td>
      <td>30</td>
    </tr>
    <tr>
      <th>LC3</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;=</td>
      <td>16</td>
    </tr>
    <tr>
      <th>LC4</th>
      <td>-1</td>
      <td>-3</td>
      <td>-4</td>
      <td>&lt;=</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LC5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>&lt;=</td>
      <td>2</td>
    </tr>
    <tr>
      <th>LC6</th>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>&lt;=</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Upper</th>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Types</th>
      <td>C</td>
      <td>C</td>
      <td>C</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>



In this section, we used the <code>ro</code> toolbox to implement a very simple linear program. In fact, the <code>ro</code> is quite versatile and can be used to model a great variety of deterministic or robust optimization problems. Further details of specifying more complicated models are provided in the next section.

## Basics

### Models

In ROAD, all optimization models are specified based on a <code>Model</code> type object. Such an object is created by the constructor <code>Model()</code> imported from the <code>road.ro</code> toolbox. 


```python
from road import ro             # Import the ro modeling tool

model = ro.Model('My model')    # Create a Model object
```

The code above defines a new <code>Model</code> object <code>model</code>, with the name specified to be <code>'My model'</code>. You could also leave the name unspecified and the default name is <code>None</code>.

### Decision variables

Decision variables of a model can be defined by the method <code>dvar()</code>.
```
Model.dvar(shape=(1,), vtype='C', name=None, aux=False) 
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
    new_var : road.lp.Vars
        An array of new decision variables
```


Similar to the <code>numpy.ndarray</code> data objects, variables in ROAD can be formatted as N-dimensional arrays, and the dimensional number is determined by the length of the tuple type attribute <code>shape</code>. Some important attributes of ROAD variables are provided below. It can be seen that they are consistent with the <code>numpy.ndarray</code> class.

```
T : Affine
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




    road.lp.Affine



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


These affine expressions can be then used in specifying the objective function of the optimization model. Please note that the objective function must be one affine expression. In other words, the <code>size</code> attribute of the expression must be one, otherwise a error message would be generated. 


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

The ROAD package also supports several convex functions for specifying convex constraints. The definition and syntax of these functions are also consistent with the NumPy package.

#### <code>abs()</code> for absolute values

The function <code>abs()</code> returns the element-wise absolute value of an array of variables or affine expressions. 

#### <code>square()</code> for squared values
The function <code>square</code> returns the element-wise squared values of an array of variables or affine expressions. 

#### <code>sumsqr()</code> for sum of squares
The function <code>sumsqr()</code> returns the sum of squares of a vector, which is a one-dimensional array, or an array with its <code>size</code> to be the same as maximum <code>shape</code> value.

#### <code>norm()</code> for norms of vectors
The function <code>sumsqr()</code> returns the first, second, or infinity norm of a vector. Users may use the second argument <code>degree</code> to specify the degree of the norm function. The default value of the <code>degree</code> argument is 2. Examples of specifying convex constraints are provided below.


```python
import road as ra
from numpy import inf

model.st(abs(z) <= 2)               # Constraints with absolute terms
model.st(ra.sumsqr(x) <= 10)        # A onstraint with sum of squares
model.st(ra.square(y) <= 5)         # Constraints with squared terms
model.st(ra.norm(z[:, 2, 0]) <= 1)  # A onstraint with 2-norm terms
model.st(ra.norm(x, 1) <= y[0, 0])  # A onstraint with 1-norm terms
model.st(ra.norm(x, inf) <= x[0])   # A Constratin with infinity norm
```

Please note that all functions above can only be used in convex functions, so convex function cannot be applied in equality constraints, and these functions cannot be used for concave inequalities, such as <code>abs(x) >= 2</code> is invalid and gives an error message.

### Standard formula and solutions

As mentioned in the previous sections, an optimization model is transformed into a standard form, which is then solved via the solver interface. The standard form can be retrieved by the <code>do_math()</code> method of the model object. 

```
Model.do_math(primal=True) 
    Returns a SOCProg type object representing the standard 
    formula as a second-order cone program. The parameter primal
    controls the returned formula is for the primal or the dual 
    problem
```

You may use the <code>do_math()</code> method together with the <code>show()</code> method to display important information on the standard formula, i.e., the objective function, linear and second-order cone constraints, bounds and variable types.


```python
import road as ra
import numpy.random as rd
from road import ro

n = 4
c = rd.normal(size=n)

model = ro.Model()
x = model.dvar(n)

model.max(c @ x)
model.st(ra.norm(x) <= 1)

primal = model.do_math()            # Formula of the primal problem
dual = model.do_math(primal=False)  # Formula of the dual problem
```

Information on the primal and dual problems are displayed below.


```python
primal.show()
```




<table border="1" class="dataframe">
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
      <th>x9</th>
      <th>x10</th>
      <th>sense</th>
      <th>constants</th>
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
      <td>0</td>
      <td>-1</td>
      <td>0</td>
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
      <td>0</td>
      <td>-1</td>
      <td>0</td>
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
      <td>0</td>
      <td>-1</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>==</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>LC5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC6</th>
      <td>-1</td>
      <td>1.30887</td>
      <td>-0.973279</td>
      <td>1.27389</td>
      <td>-2.64835</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>&lt;=</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Upper</th>
      <td>inf</td>
      <td>inf</td>
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
      <th>Lower</th>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Types</th>
      <td>C</td>
      <td>C</td>
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




```python
dual.show()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>sense</th>
      <th>constants</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Obj</th>
      <td>0</td>
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
      <td>0</td>
      <td>1.30887</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.973279</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.27389</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LC5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-2.64835</td>
      <td>==</td>
      <td>1</td>
    </tr>
    <tr>
      <th>QC1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>&lt;=</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Upper</th>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Types</th>
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



### Application examples

#### Mean-variance portfolio optimization

In this example, we consider a portfolio construction problem discussed in [Robust Solutions of Uncertain Linear Programs](https://www2.isye.gatech.edu/~nemirovs/stablpn.pdf). Suppose there are <img src="https://latex.codecogs.com/gif.latex?n=150 " />  stocks, and each stock <img src="https://render.githubusercontent.com/render/math?math=i"> has a random return <img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7Bp%7D_i">. Let <img src="https://render.githubusercontent.com/render/math?math=x_i"> be the fraction of wealth invested in stock <img src="https://render.githubusercontent.com/render/math?math=i">, the objective of the portfolio problem is to maximize the total return <img src="https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5En%5Ctilde%7Bp%7D_ix_i">. It is assumed that each stock <img src="https://render.githubusercontent.com/render/math?math=i"> is independent, and it has the mean return to be <img src="https://render.githubusercontent.com/render/math?math=p_i"> and the standard deviation is <img src="https://render.githubusercontent.com/render/math?math=%5Csigma_i">. A classic approach is to formulate the problem as a quadratic program, where a mean-variance objective function is maximized:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign%7D%0A%5Cmax%26%5Csum%5Climits_%7Bi%3D1%7D%5Enp_ix_i%20-%20%5Cphi%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csigma_i%5E2x_i%5E2%20%5C%5C%0A%5Ctext%7Bs.t.%20%7D%26%5Csum%5Climits_%7Bi%3D1%7D%5Enx_i%20%3D%201%20%5C%5C%0A%26%20x_i%20%5Cgeq%201%2C%20%5Cforall%20i%20%3D%201%2C%202%2C%20...%2C%20n%2C%0A%5Cend%7Balign%7D">

with the constant $\phi=5$, and the means and standard deviations are specified to be
\begin{align}
&p_i = 1.15 + i\frac{0.05}{150} \\
&\sigma_i = \frac{0.05}{450}\sqrt{2in(n+1)}.
\end{align}

The quadratic program can be implemented by the following code segment. 


```python
import road as ra
import numpy as np
from road import ro
from road import grb_solver as grb

n = 150                                 # Number of stocks
i = np.arange(1, n+1)                   # Indices of stocks
p = 1.15 + i*0.05/150                   # Mean returns
sigma = 0.05/450 * (2*i*n*(n+1))**0.5   # S.T.D. of returns
phi = 5                                 # Constant phi

model = ro.Model('mv-portfolio')

x = model.dvar(n)                       # Fractions of investment

model.max(p@x - phi*ra.sumsqr(sigma*x)) # Mean-variance objective
model.st(x.sum() == 1)                  # Summation of x is one
model.st(x >= 0)                        # x is non-negative

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


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_11_0.png">


    Objective value: 1.1853


#### Integer programming for Sudoku

In this section we will use a [Sudoku](https://en.wikipedia.org/wiki/Sudoku) game to illustrate how to use integer and multi-dimensional arrays in ROAD. Sudoku is a popular number puzzle. The goal is to place the digits in \[1,9\] on a nine-by-nine grid, with some of the digits already filled in. Your solution must satisfy the following four rules:

1. Each cell contains an integer in \[1,9\].
2. Each row must contain each of the integers in \[1,9\].
3. Each column must contain each of the integers in \[1,9\].
4. Each of the nine 3x3 squares with bold outlines must contain each of the integers in \[1,9\].

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg/1280px-Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg.png" width=200>

The Sudoku game can be considered as a optimization with the objective to be zero and constraints used to fulfill above rules. Consider a binary variable $x_{ijk}\in \{0, 1\}$, where $\forall i \in [0, 8]$, $j \in [0, 8]$, $k \in [0, 8]$. It equals to one if a integer $k+1$ is placed in a cell at the $i$th row and $j$th column. Let $a_{ij}$ be the known number at the $i$th row and $j$th column, where $i\in\mathcal{I}$ and $j\in\mathcal{J}$, the Sudoku game can be written as the following integer programming problem
\begin{align}
\min~&0 \\
\text{s.t.}~& \sum\limits_{i=0}^8x_{ijk} = 1, \forall j \in [0, 8], k \in [0, 8] \\
& \sum\limits_{j=0}^8x_{ijk} = 1, \forall i \in [0, 8], k \in [0, 8] \\
& \sum\limits_{k=0}^8x_{ijk} = 1, \forall i \in [0, 8], j \in [0, 8] \\
& x_{ij(a_{ij}-1)} = 1, \forall i \in \mathcal{I}, j \in \mathcal{J} \\
& \sum\limits_{m=0}^2\sum\limits_{n=0}^2x_{(i+m), (j+m), k} = 1, \forall i \in \{0, 3, 6\}, j \in \{0, 3, 6\}, k \in [0, 8]
\end{align}

In the following code, we are using ROAD to implement such a model. 


```python
import road as ra
import numpy as np
from road import ro
from road import grb_solver as grb

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
for i in range(9):
    for j in range(9):
        a_ij = puzzle[i, j] 
        if a_ij > 0:
            model.st(x[i, j, a_ij -1] == 1)

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


The binary variable $x_{ijk}$ is defined to be a three-dimensional array <code>x</code> with the shape to be <code>(9, 9, 9)</code>. Please note that in ROAD, the objective function cannot be specified as a numeric constant, we then use the expression <code>0 * x.sum()</code> as the objective. Based on the decision variable <code>x</code>, each set of constraints can be formulated as the array form by using the <code>sum</code> method. The method <code>sum()</code> in ROAD is consistent with that in NumPy, where you may use the <code>axis</code> argument to specify along which axis the sum is performed. 

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


## Robust optimization

The ROAD package is capable of formulating the robust optimization problems as
\begin{align}
\min\limits_{\pmb{x}} ~ &\max\limits_{\pmb{z}\in\mathcal{Z}_0} ~\pmb{z}^T\pmb{A}_0\pmb{x} + \pmb{b}_0^T\pmb{x} + \pmb{c}_0^T\pmb{z} + d_0 &&\\
\text{s.t.}~&\pmb{z}^T\pmb{A}_1\pmb{x}+ \pmb{b}_1^T\pmb{x} + \pmb{c}_1^T\pmb{z} + d_1 \leq 0, &&\forall \pmb{z}\in\mathcal{Z}_1 \\
&\pmb{z}^T\pmb{A}_2\pmb{x}+ \pmb{b}_2^T\pmb{x} + \pmb{c}_2^T\pmb{z} + d_2 \leq 0, &&\forall \pmb{z}\in\mathcal{Z}_2 \\
& \vdots \\
&\pmb{z}^T\pmb{A}_m\pmb{x}+ \pmb{b}_m^T\pmb{x} + \pmb{c}_m^T\pmb{z} + d_m \leq 0, &&\forall \pmb{z}\in\mathcal{Z}_m \\
& \pmb{x} \in \mathcal{X}.
\end{align}

where $\pmb{x}$ is the array decision variables, and $\pmb{z}$ is the array of random variables. The generic formula above suggests that we are minimizing (or maximizing) the objective function under the worst-case realization of the uncertainty set $\mathcal{Z}_0$, subject to the worst-case constraints over each uncertainty set $\mathcal{Z}_j$, $\forall j \in 1, 2, ..., m$.

The ROAD package provides rich algebraic tools for specifying the random variables arrays, uncertainty sets, the worst-case objective and constraints of the robust model.

### Random variables

Random variables of a robust optimization model can be defined by the method <code>rvar()</code> of the model object. 

```
Model.rvar(shape=(1,), name=None) 
    Returns an array of decision variables with the given shape 
    and variable type.
    
    Parameters
    ----------
    shape : int or tuple
        Shape of the variable array.
    name : str
        Name of the variable array
    
    Returns
    -------
    new_var : road.lp.Vars
        An array of new random variables
```

Similar to decision variables, random variables are also formulated as arrays, and all array operations and functions aforementioned could be applied to random variables.


```python
from road import ro

model = ro.Model()          # Create a model object
x = model.dvar((1, 5))      # A 1x5 array of decision varaibles
y = model.dvar((2, 1))      # A 2x1 array of decision variables
z = model.rvar((2, 5))      # A 2x5 array of random variables

model.st(x * z <= 2)        # Multiplication with broadcasting
model.st(y.T@z - x <= 5)    # Matrix multiplication
```

### Uncertainty sets

The uncertainty set $\mathcal{Z}_0$ for the objective function can be specified by the method <code>minmax()</code> and <code>maxmin()</code>. Take the following uncertainty set $\mathcal{Z}_0$ for example, 
\begin{align}
\mathcal{Z}_0 = \left\{\pmb{z}: 
\|\pmb{z}\|_{\infty} \leq 1, 
\|\pmb{z}\|_1 \leq 1.5  
\right\}
\end{align}


```python
from road import ro
import road as ra

model = ro.Model()          
x = model.dvar((2, 5))      
z = model.rvar((1, 5))    

# Define uncertainty set Z0 as a tuple
z_set0 = (ra.norm(z, np.inf) <= 1,   
          ra.norm(z, 1) <= 1.5)

# Minimize the worst-case objective over the uncertainty set Z0
model.minmax((x*z).sum(), z_set0)    

# Maximize the worst-case objective over the uncertainty set Z0
model.maxmin((x*z).sum(), z_set0)
```

In the functions <code>minmax()</code> and <code>maxmin()</code>, the first argument is the objective function, and all the remaining arguments are used to specify the constraints of the uncertainty set $\mathcal{Z}_0$.

For robust constraints, the uncertainty set can be specified by the <code>forall()</code> method, as shown by the following example. 


```python
# Define uncertainty set Z1 as a tuple
z_set1 = (ra.norm(z, np.inf) <= 1.5,
          ra.norm(z, 1) <= 2)

# The constraints over the uncertainty set Z1
model.st((x*z + z >= 0).forall(z_set1))
```

Please note that if the uncertainty set of a robust constraint is not defined, then its uncertainty set is $\mathcal{Z}_0$, defined for the worst-case objective. The code below therefore considers one uncertainty set $\mathcal{Z}_0$ for the objective and all constraints. 


```python
from road import ro
import road as ra

model = ro.Model()          
x = model.dvar((2, 5))      
z = model.rvar((1, 5))    

# Define uncertainty set Z0 as a tuple
z_set0 = (ra.norm(z, np.inf) <= 1,   
          ra.norm(z, 1) <= 1.5)

# Define objective function and the uncertainty set
model.minmax((x*z).sum(), z_set0)  

# The uncertainty set Z0 applies to all constraints below
model.st(x*z + z >= 0)
model.st(x*z + x >= 0)
model.st(x >= z)
```

### Application examples

#### Robust portfolio optimization

In this example, the portfolio construction problem discussed in the previous sections is solved by a robust optimization approach introduced in the paper [The Price of Robustness](https://www.researchgate.net/publication/220244391_The_Price_of_Robustness). The robust model is presented below.

\begin{align}
\max~&\min\limits_{\pmb{z}\in\mathcal{Z}} \sum\limits_{i=1}^n\left(p_i + \delta_iz_i \right)x_i \\
\text{s.t.}~&\sum\limits_{i=1}^nx_i = 1 \\
&x_i \geq 0,
\end{align}
where the affine term $p_i + \delta_iz_i$ represents the random stock return, and the random variable is between $[-1, 1]$, so the stock return has an arbitrary distribution in the interval $[p_i - \delta_i, p_i + \delta_i]$. The uncertainty set $\mathcal{Z}$ is given below,
\begin{align}
\mathcal{Z} = \left\{\pmb{z}: \|\pmb{z}\|_{\infty} \leq 1, \|\pmb{z}\|_1 \leq \Gamma\right\},
\end{align}
where $\Gamma$ is the budget of uncertainty parameter. Values of the budget of uncertainty and other parameters are presented as follows.
\begin{align}
& \Gamma = 3 \\
& p_i = 1.15 + i\frac{0.05}{150} \\
& \delta_i = \frac{0.05}{450}\sqrt{2in(n+1)}.
\end{align}

The robust optimization model can be implemented by the following Python code.


```python
import road.ro as ro
import road as ra
import numpy as np

n = 150                                 # Number of stocks
i = np.arange(1, n+1)                   # Indices of stocks
p = 1.15 + i*0.05/150                   # Mean returns
delta = 0.05/450 * (2*i*n*(n+1))**0.5   # Deviations of returns
Gamma = 3                               # Budget of uncertainty

model = ro.Model()              
x = model.dvar(n)                       # Fractions of investment
z = model.rvar(n)                       # Random variables

model.maxmin((p + delta*z) @ x,         # The max-min objective
             ra.norm(z, np.infty) <=1,  # Uncertainty set constraints
             ra.norm(z, 1) <= Gamma)    # Uncertainty set constraints
model.st(sum(x) <= 1)                   # Summation of x is one
model.st(x >= 0)                        # x is non-negative

model.solve(grb)                        # Solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0031s


The optimal investment decision can be visualized by the diagram below.


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


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_17_0.png" >


    Objective value: 1.1709


#### The robust and robustness knapsack problem

In this example, we will use the ROAD package to implement the robust knapsack model introduced in the paper [The Price of Robustness](https://www.researchgate.net/publication/220244391_The_Price_of_Robustness), and the robustness model described in the paper [The Dao of Robustness](http://www.optimization-online.org/DB_FILE/2019/11/7456.pdf). The robust model can be written as
\begin{align}
\max~&\sum\limits_{i=1}^nc_i x_i \\
\text{s.t.}~&\sum\limits_{i=1}^n(\hat{w}_i + z_i\delta_i)x_i \leq b, \forall \pmb{z} \in \mathcal{Z} \\
&\pmb{x} \in \{0, 1\}^n
\end{align}
with the random variable $\pmb{z}$ constrained by an uncertainty set 
\begin{align}
\mathcal{Z}=\left\{\pmb{z}: \|\pmb{z}\|_{\infty} \leq 1, \|\pmb{z}\|_{1} \leq r\right\}, 
\end{align}
where the parameter $r$ is the budget of uncertainty. The robustness optimization model can be formulated by introducing an auxiliary random variable $\pmb{u}$, such that
\begin{align}
\min~&k\\
\text{s.t.}~&\sum\limits_{i=1}^nc_i x_i \geq \Gamma \\
&\sum\limits_{i=1}^n(\hat{w}_i + z_i\delta_i)x_i - b \leq k \sum\limits_{i=1}^nu_i, \forall (\pmb{z}, \pmb{u}) \in \overline{\mathcal{Z}} \\
&\pmb{x} \in \{0, 1\}^n
\end{align}
with $\Gamma$ to a target of the objective value, and the random variables $\pmb{z}$ and $\pmb{u}$ constrained by an lifted uncertainty set 
\begin{align}
\overline{\mathcal{Z}}=\left\{(\pmb{z}, \pmb{u}): |z_i|\leq u_i\leq 1, \forall i=1, 2, ..., n \right\}.
\end{align}

Following the aforementioned papers, parameters $\pmb{c}$, $\hat{\pmb{w}}$, and $\pmb{\delta}$ are randomly generated by the code below.


```python
import road.ro as ro
import road.grb_solver as grb
import road as ra
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

N = 50 
b = 2000

c = 2*rd.randint(low=5, high=10, size=N)        # Profit coefficients
w_hat = 2*rd.randint(low=10, high=41, size=N)   # Nominal weights
delta = 0.2*w_hat                               # Maximum deviations
```

The robust optimization model for a given budget of uncertainty $r$ is written as a function named <code>robust()</code>.


```python
def robust(r):
    """
    The function robust implements the robust optmization model, 
    given the budget of uncertainty r
    """

    model = ro.Model('robust')
    x = model.dvar(N, vtype='B')    
    z = model.rvar(N)              

    z_set = (abs(z) <= 1, ra.norm(z, 1) <= r)
    model.max(c @ x)
    model.st(((w_hat + z*delta) @ x <= b).forall(z_set))

    model.solve(grb, display=False) # Disable solution message
    
    return model.get(), x.get()     
```

Similarly, the robustness optimization model for a given target of profit $\Gamma$ can also be written as a function <code>robustness()</code>.


```python
def robustness(Gamma):
    """
    The function robustness implements the robustness optmization 
    model, given the profit target Gamma.
    """
    
    model = ro.Model('robustness')

    x = model.dvar(N, vtype='B')    
    k = model.dvar()              
    z = model.rvar(N)           
    u = model.rvar(N)           

    z_set = (abs(z) <= u, u <= 1)
    model.min(k)
    model.st(c @ x >= Gamma)
    model.st(((w_hat + z*delta) @ x - b <= k*u.sum()).forall(z_set))
    model.st(k >= 0)
    
    model.solve(grb, display=False) # Disable solution message
    
    return model.get(), x.get()
```

Given a decision $\pmb{x}$ and a sample of the random variable $\pmb{z}$, we write a function <code>sim()</code> to calculate the probability of constraint violation, as an indicator of the performance of solutions. 


```python
def sim(x_sol, zs):
    """
    The function sim is for calculating the probability of violation 
    via simulations. 
        x_sol: solution of the Knapsack problem
        zs: random sample of the random variable z
    """
    
    ws = w_hat + zs*delta
    
    return (ws @ x_sol > b).mean()

```

By using functions above, we run the robust and robustness optimization models and assess their performance via simulations. 


```python
step = 0.1
rs = np.arange(1, 5+step, step)         # All budgets of uncertainty
num_samp = 20000
zs = 1-2*rd.rand(num_samp, N)           # Random samples for z

"""Robust optimization"""
outputs_rb = [robust(r) for r in rs]
tgts = [output[0] 
        for output in outputs_rb]       # Objective used as targets
pv_rb = [sim(output[1], zs) 
         for output in outputs_rb]      # Prob. of violations 

"""Robustness optimization"""
outputs_rbn = [robustness(tgt) 
               for tgt in tgts]   
pv_rbn = [sim(output[1], zs) 
          for output in outputs_rbn]    # Prob. of violations 
```

The probabilities of violations for both methods are visualized by the following diagrams. 


```python
plt.plot(rs, pv_rb, marker='o', markersize=5, c='b', 
         label='Robust Optimization')
plt.plot(rs, pv_rbn, c='r', 
         label='Robustness Optimization')

plt.legend()
plt.xlabel('Parameter r in robust optimization')
plt.ylabel('Prob. violation')
plt.show()

plt.scatter(tgts, pv_rb, c='b', alpha=0.3,
            label='Robust Optimization')
plt.scatter(tgts, pv_rbn, c='r', alpha=0.3,
            label='Robustness Optimization')

plt.legend()
plt.xlabel(r'Target return $\tau$')
plt.ylabel('Prob. violation')
plt.show()
```


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_35_0.png">

<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_35_1.png">


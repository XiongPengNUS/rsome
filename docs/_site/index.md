<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Introduction

ROAD (Robust Optimization for Array Data) is an open-source Python package for operations research and generic optimization modeling. ROAD models are constructed by variables, expressions, and constraints formatted as N-dimensional arrays, which are consistent with the NumPy library in syntax and operations, such as indexing and slicing, element-wise operations, broadcasting, and matrix calculation rules. It thus provides a convenient and highly readable way in developing optimization models and applications.

The current version of ROAD supports deterministic linear/second-order cone programs and robust optimization problems. An interface with the Gurobi solver is also integrated for the solution of optimization models. Distributionally robust optimization modeling tools based on the [robust stochastic optimization (RSO) framework](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603?af=R) is now under development. Other solver interfaces will be included in the future.

### Installing ROAD and solvers

The ROAD package can be installed with the <code>pip</code> command:

***

`pip install road`

***

For the current version, the Gurobi solver is also needed for solving the optimization model, and you may follow [these steps](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html) to complete the solver installation.

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

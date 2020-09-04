<img src="https://github.com/XiongPengNUS/rsome/blob/master/rsologo.png?raw=true" width=100>

# RSOME: Robust Stochastic Optimization Made Easy

RSOME (Robust Stochastic Optimization Made Easy) is an open-source Python package for generic modeling optimization problems. Models in RSOME are constructed by variables, constraints, and expressions that are formatted as N-dimensional arrays. These arrays are consistent with the NumPy library in terms of syntax and operations, including broadcasting, indexing, slicing, element-wise operations, and matrix calculation rules, among others. In short, RSOME provides a convenient platform to facilitate developments of optimization models and their applications.

The current version of RSOME supports deterministic linear/second-order cone programs and robust optimization problems. Interfaces with Gurobi and MOSEK solvers are integrated for solving the optimization models in RSOME. Distributionally robust optimization modeling tools based on the [robust stochastic optimization (RSO) framework](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603?af=R) and interfaces with other solvers are under development.

## Introduction

### Installing RSOME and solvers

The RSOME package can be installed by using the <code>pip</code> command:
***
**`pip install rsome`**
***

The current version of RSOME requires the Gurobi or MOSEK solver for solving the formatted models. You may follow [these steps](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html) to complete Gurobi installation. The MOSEK solver can be installed via steps in [this link](https://docs.mosek.com/9.2/pythonapi/install-interface.html).


### Getting started

In RSOME, models can be specified by using highly readable algebraic expressions that are consistent with NumPy syntax. Below we provide a simple linear program as an example to illustrate the steps of modeling and solving an optimization problem.


![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cbg_white%20%5Cbegin%7Balign%7D%20%5Cmax%20%7E%263x%20&plus;%204y%20%5Cnonumber%20%5C%5C%20%5Ctext%7Bs.t.%7D%7E%262.5x%20&plus;%20y%20%5Cleq%2020%20%5Cnonumber%20%5C%5C%20%265x%20&plus;%203y%20%5Cleq%2030%20%5Cnonumber%20%5C%5C%20%26x%20&plus;%202y%20%5Cleq%2016%20%5Cnonumber%20%5C%5C%20%26%7Cy%7C%20%5Cleq%202%2C%20%5Cnonumber%20%5Cend%7Balign%7D)


```python
from rsome import ro                 # Import the ro modeling tool
from rsome import grb_solver as grb  # Import Gurobi solver interface

model = ro.Model('LP model')         # Create a Model object
x = model.dvar()                     # Define a decision variable x
y = model.dvar()                     # Define a decision variable y

model.max(3*x + 4*y)                 # Maximize the objective function
model.st(2.5*x + y <= 20)            # Specify the 1st constraint
model.st(5*x + 3*y <= 30)            # Specify the 2nd constraint
model.st(x + 2*y <= 16)              # Specify the 3rd constraint
model.st(abs(y) <= 2)                # Specify the 4th constraint

model.solve(grb)                     # Solve the model with Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0005s


In this sample code, a model object is created by calling the constructor <code>Model()</code> imported from the <code>rsome.ro</code> toolbox. Based on the model object, decision variables <code>x</code> and <code>y</code> are created with the method <code>dvar()</code>. Variables are then used in specifying the objective function and constraints. The last step is to call the <code>solve()</code> method to solve the problem via the imported solver interface <code>grb</code>. Once the solution procedure completes, a message showing the solution status and running time will be printed.

According to the [Gurobi solution status](https://www.gurobi.com/documentation/9.0/refman/optimization_status_codes.html), the status code <code>2</code> suggests that the problem was solved to optimality (subject to tolerances) and an optimal solution is available. The optimal solution and the corresponding objective value can be obtained by the <code>get()</code> method.


```python
print('x:', x.get())
print('y:', y.get())
print('Objective:', model.get())
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

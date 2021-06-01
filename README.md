<img src="https://github.com/XiongPengNUS/rsome/blob/master/rsologo.png?raw=true" width=100>

# RSOME: Robust Stochastic Optimization Made Easy
- Website: [RSOME for Python](https://xiongpengnus.github.io/rsome/)
- PyPI: [RSOME 0.1.3](https://pypi.org/project/rsome/)

RSOME (Robust Stochastic Optimization Made Easy) is an open-source Python package for generic modeling of optimization problems (subject to uncertainty). Models in RSOME are constructed by variables, constraints, and expressions that are formatted as N-dimensional arrays. These arrays are consistent with the NumPy library in terms of syntax and operations, including broadcasting, indexing, slicing, element-wise operations, and matrix calculation rules, among others. In short, RSOME provides a convenient platform to facilitate developments of optimization models and their applications.

The current version of RSOME supports deterministic, robust optimization and distributionally robust optimization problems. In the default configuration, linear programs are solved by the open-source solver `linprog()` imported from the `scipy.optimize` package. Details of this solver interface, together with interfaces of other open-source and commercial solvers are presented in the following table.

| Solver | License  type | RSOME interface |Integer variables| Second-order cone constraints|
|:-------|:--------------|:----------------|:------------------------|:---------------------|
|[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)| Open-source | `lpg_solver` | No | No |
|[CyLP](https://github.com/coin-or/cylp)| Open-source | `clp_solver` | Yes | No |
|[OR-Tools](https://developers.google.com/optimization/install) | Open-source | `ort_solver` | Yes | No |
|[Gurobi](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html)| Commercial | `grb_solver` | Yes | Yes |
|[MOSEK](https://docs.mosek.com/9.2/pythonapi/install-interface.html) | Commercial | `msk_solver` | Yes | Yes |
|[CPLEX](https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) | Commercial | `cpx_solver` | Yes | Yes |

## Introduction

### Installing RSOME and solvers

The RSOME package can be installed by using the <code>pip</code> command:
***
**`pip install rsome`**
***


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


The example above shows that RSOME models can be formulated via straightforward and highly readable algebraic expressions, and the reformulation can be transformed into a standard form, which is then solved by the Gurobi (or MOSEK) solver. The basic information of the standard form can be retrieved by calling the <code>do_math()</code> method of the RSOME model object.


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

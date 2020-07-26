# ROAD

ROAD, as Robust Optimization with Array-like Data, is an open-source toolbox designed for implementing and prototyping robust optimization models. The package is consistent with the NumPy package in terms of indexing, N-dimensional array operations, and matrix calculation rules, and it provides a convenient interface in specifying model features. The constructed model can be automatically converted into a counterpart formulation, which can be solved by integrated commercial solvers, such as Gurobi. 

## The Dao of ROAD


```python
from road import dao
```


The Dao of ROAD: 



[ROME](https://robustopt.com/) was not built in one day. 



All [ROAD](https://github.com/XiongPengNUS/ROAD)s lead to [ROME](https://robustopt.com/).



Matlab is [RSOME](https://www.rsomerso.com/). 



The [ROAD](https://github.com/XiongPengNUS/ROAD) in Python is more than [RSOME](https://www.rsomerso.com/).


## Examples

### Linear Programming
We are using a very simple linear program to demonstrate the general procedure of running <code>ROAD</code> for modeling optimization problems.


```python
import road.lp as lp               # Import the LP modeling module from ROAD
import road.grb_solver as grb      # Import the solver interface for Gurobi

model = lp.Model()                  # Create an LP model
x = model.dvar()                    # Define a decision variable x
y = model.dvar()                    # Define a decision variable y

model.max(3*x + 4*y)                # Maximize the objective function
model.st(2.5*x + y <= 20)           # Define constriants
model.st(3*x + 3*y <= 30)
model.st(x + 2*y <= 16)
model.st(x <= 3)
model.st(abs(y) <= 2)

model.solve(grb)                    # Solve the model by Gurobi
```

    Using license file /Users/pengxiong/gurobi.lic
    Academic license - for non-commercial use only
    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0003s


The optimal objective value and optimal solutions can be retrieved by the <code>get()</code> method of the <code>model</code> object or the variable object.


```python
print(model.get())
print(x.get())
print(y.get())
```

    17.0
    [3.]
    [2.]



```python
model.do_math()
```




    Linear program object:
    =============================================
    Number of variables:          3
    Continuous/binaries/integers: 3/0/0
    ---------------------------------------------
    Number of linear constraints: 6
    Inequalities/equalities:      6/0
    Number of coefficients:       11



Please note that the optimal solutions of the decision variables are given as <code>array</code> type objects. To facilitate debugging, the <code>ROAD</code> package could generate a table of constraint information via running the command <code>model.do_math().showlc()</code>.  


```python
model.do_math().showlc()
```




 <b> </b> |<b>x1</b>|<b>x2</b>|<b>x3</b>|<b>sense</b>|<b>constants</b>
---:|------:|----------:|---------:|----------:|---------:
<b>LC1</b> | 0.0 | 2.5 | 1.0 | <= | 20.0
<b>LC2</b> | 0.0 | 3.0 | 3.0 | <= | 30.0
<b>LC3</b> | 0.0 | 1.0 | 2.0 | <= | 16.0
<b>LC4</b> | -1.0 | 3.0 | -4.0 | <= | 0.0
<b>LC5</b> | 0.0 | 0.0 | 1.0 | <= | 2.0
<b>LC6</b> | 0.0 | 0.0 | -1.0 | <= | 2.0




### Mean-variance portfolio optimization
In this example, we formulate the mean-variance portfolio optimization problem as a second-order cone program. Details of this model can be found from the paper [Price of Robustness](https://www.robustopt.com/references/Price%20of%20Robustness.pdf).


```python
import road as ra                         # Import the ROAD package
import road.socp as cp                    # Import the SOCP modeling module from ROAD
import road.grb_solver as grb             # Import the solver interface for Gurobi
import numpy as np

n = 150                                     # Number of stocks
i = np.arange(1, n+1)                   
p = 1.15 + 0.05/n*i                         # Mean values of stock returns
sig = 0.05/450 * (2*i*n*(n+1))**0.5         # Standard devaition of stock returns
phi = 5                                     # Constant phi

model = cp.Model()                          # Create an SOCP model
x = model.dvar(n)                           # Define decision variables as an array with length n

model.max(p@x - phi*ra.sumsqr(sig * x))     # Define the objective function
model.st(sum(x) == 1)                       # Define the constriants
model.st(x >= 0)

model.solve(grb)                            # Solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0023s


The solution in terms of the optimal stock allocation is presented below.


```python
import matplotlib.pyplot as plt

plt.plot(i, x.get())
plt.xlabel('stocks')
plt.ylabel('x')
plt.show()
```


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_11_0.png">


### Robust portfolio optimization
The robust portfolio optimization model introduced in the paper [Price of Robustness](https://www.robustopt.com/references/Price%20of%20Robustness.pdf) can also be formulated by the <code>ROAD</code> package.


```python
import road.ro as ro                                # Import the robust optimization module from ROAD

n = 150                                             # Number of stocks
i = np.arange(1, n+1)
p = 1.15 + 0.05/n*i                                 # Mean values of stock returns
sig = 0.05/450 * (2*i*n*(n+1))**0.5                 # Maximum deviation of stock returns
Gamma = 3                                           # Budget of uncertainty

model = ro.Model()                                  # Create a robust optimization model
x = model.dvar(n)                                   # Define decision variables x as an array
z = model.rvar(n)                                   # Define random variables z as an array

model.maxmin((p + sig*z) @ x,                       # Define the max-min objective function
             abs(z)<=1, ra.norm(z, 1) <= Gamma)     # Uncertainty set of the model
model.st(sum(x) <= 1)                               # Define constraints
model.st(x >= 0)

model.solve(grb)                                    # Solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0031s


The solution as the stock allocation is given below.


```python
plt.plot(i, x.get())
plt.xlabel('stocks')
plt.ylabel('x')
plt.show()
```


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_17_0.png" width=450>


### Robust optimization for a lot-sizing problem
The robust model for the lot-sizing problem described in the paper [Duality in Two-Stage Adaptive Linear Optimization: Faster Computation and Stronger Bounds](https://www.fransderuiter.com/papers/BertsimasdeRuiter2016.pdf) is implemented with <code>PyAtom</code> to demonstrate how the linear decision rules is specified for adaptive decisions. 

The parameters for a $N=30$ case are defined below.


```python
from road import ro
import road.grb_solver as grb
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

N = 30
c = 20
dmax = 20
Gamma = dmax*np.sqrt(N)
xy = 10*rd.rand(2, N)
tmat = ((xy[[0]] - xy[[0]].T) ** 2 + 
        (xy[[1]] - xy[[1]].T) ** 2) ** 0.5
```

The model is then implemented by the following code.


```python
model = ro.Model()

d = model.rvar(N)

x = model.dvar(N)

y = model.ldr((N, N))
y.adapt(d)

model.minmax((c*x).sum() + (tmat*y).sum(), 
             (d >= 0, d <= dmax, sum(d) <= Gamma))
model.st(d <= y.sum(axis=0) - y.sum(axis=1) + x)
model.st(y >= 0)
model.st(x >= 0)
model.st(x <= 20)

model.solve(grb)
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.3973s


The optimal solution of the stock allocation <code>x</code> can be retrieved by calling the method <code>x.get()</code>, and the results are visualized as follows.


```python
plt.figure(figsize=(5, 5))
plt.scatter(xy[0], xy[1], c='w', edgecolors='k')
plt.scatter(xy[0], xy[1], s=50*x.get(), c='k', alpha=0.4)
plt.axis('equal')
plt.xlim([-1, 11])
plt.ylim([-1, 11])
plt.grid()
plt.show()
```


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_23_0.png" width=450>



### A Knapsack problem: robust optimization v.s. robustness optimization

In this example, we use the <code>ROAD</code> package to implement the robust and robustness optimization models described in the paper [The Dao of Robustness](http://www.optimization-online.org/DB_FILE/2019/11/7456.pdf)


```python
import road.ro as ro
import road.grb_solver as grb
import road as ra
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

N = 50 
b = 2000

c = 2*rd.randint(low=5, high=10, size=N)        # Profit coefficients of 
w_hat = 2*rd.randint(low=10, high=41, size=N)   # Baseline values of the weights

delta = 0.2*w_hat                               # Maximum deviations
```

The function for the robust optimization method is given below.


```python
def robust(r):
    """
    The function robust implements the robust optmization model, given the budget of 
    uncertainty r
    """

    model = ro.Model('robust')
    x = model.dvar(N, vtype='B')        # Define decision variables
    z = model.rvar(N)                   # Define random variables

    model.max(c @ x)
    model.st(((w_hat + z*delta) @ x <= b).forall(abs(z) <= 1, ra.norm(z, 1) <= r))

    model.solve(grb, display=False)
    
    return model.get(), x.get()         # Return objective value and the optimal solution
```

The function for the robustness optimization model.


```python
def robustness(target):
    
    model = ro.Model('robustness')

    x = model.dvar(N, vtype='B')
    k = model.dvar()
    z = model.rvar(N)
    u = model.rvar(N)

    model.min(k)
    model.st(c @ x >= target)
    model.st(((w_hat + z*delta) @ x - b <= k*u.sum()).forall(abs(z) <= u, u <= 1))
    model.st(k >= 0)
    
    model.solve(grb, display=False)
    
    return model.get(), x.get()
```

The following function <code>sim</code> is for calculating the probability of violation via simulations. 


```python
def sim(x_sol, zs):
    """
    The function sim is for calculating the probability of violation via simulations. 
        x_sol: solution of the Knapsack problem
        zs: random sample of the random variable z
    """
    
    ws = w_hat + zs*delta
    
    return (ws @ x_sol > b).mean()
```

By using functions above we then run the robust and robustness optimization models.


```python
step = 0.1
rs = np.arange(1, 5+step, step)                 # All budgets of uncertainty
num_samp = 20000
zs = 1-2*rd.rand(num_samp, N)                   # Random samples for z

"""Robust optimization"""
outputs_rb = [robust(r) for r in rs]            # Robust optimization models
tgts = [output[0] for output in outputs_rb]     # Objective values as the targets
pv_rb = [sim(output[1], zs) 
         for output in outputs_rb]              # Prob. violation for robust optimization

"""Robustness optimization"""
outputs_rbn = [robustness(tgt) 
               for tgt in tgts]                 # Robustness optimization models
pv_rbn = [sim(output[1], zs) 
          for output in outputs_rbn]            # Objective values as the targets
```

The results are visualized as follows.


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


<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_35_0.png" width=450>

<img src="https://raw.githubusercontent.com/XiongPengNUS/road/master/output_35_1.png" width=450>


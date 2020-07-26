# PyAtom

PyAtom is an open-source Python Algebraic Toolbox for Optimization Modeling. It is consistent with the NumPy package in terms of N-dimensional array operations, and relies on the commercial solver Gurobi to solve the formatted models. So far, the toolbox is capable of modeling robust optimization problems formatted as second-order cone programs.

## Examples

### Linear Programming
We are using a very simple linear program to demonstrate the general procedure of running <code>PyAtom</code> for modeling optimization problems.


```python
import pyatom.lp as lp              # Import the LP modeling module from PyAtom
import pyatom.grb_solver as grb     # Import the solver interface for Gurobi

model = lp.Model()                  # Create an LP model
x = model.dvar()                    # Define a decision variable x
y = model.dvar()                    # Define a decision variable y

model.max(3*x + 4*y)                # Maximize the objective function
model.st(2.5*x + y <= 20)           # Define constraints
model.st(3*x + 3*y <= 30)
model.st(x + 2*y <= 16)
model.st(x <= 3)
model.st(abs(y) <= 2)

model.solve(grb)                    # Solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0002s


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



Please note that the optimal solutions of the decision variables are given as <code>array</code> type objects. To facilitate debugging, the <code>PyAtom</code> package could generate a table of constraint information via running the command <code>model.do_math().showlc()</code>.  


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
import pyatom as at                         # Import the PyAtom package
import pyatom.socp as cp                    # Import the SOCP modeling module from PyAtom
import pyatom.grb_solver as grb             # Import the solver interface for Gurobi
import numpy as np

n = 150                                     # Number of stocks
i = np.arange(1, n+1)                   
p = 1.15 + 0.05/n*i                         # Mean values of stock returns
sig = 0.05/450 * (2*i*n*(n+1))**0.5         # Standard deviation of stock returns
phi = 5                                     # Constant phi

model = cp.Model()                          # Create an SOCP model
x = model.dvar(n)                           # Define decision variables as an array with length n

model.max(p@x - phi*at.sumsqr(sig * x))     # Define the objective function
model.st(sum(x) == 1)                       # Define the constraints
model.st(x >= 0)

model.solve(grb)                            # Solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0024s


The solution in terms of the optimal stock allocation is presented below.


```python
import matplotlib.pyplot as plt

plt.plot(i, x.get())
plt.xlabel('stocks')
plt.ylabel('x')
plt.show()
```


<img src='https://github.com/XiongPengNUS/pyatom/blob/master/output_11_0.png?raw=true' width=500pt>


### Robust portfolio optimization
The robust portfolio optimization model introduced in the paper [Price of Robustness](https://www.robustopt.com/references/Price%20of%20Robustness.pdf) can also be formulated by the <code>PyAtom</code> package.


```python
import pyatom.ro as ro                              # Import the robust optimization module from PyAtom

n = 150                                             # Number of stocks
i = np.arange(1, n+1)
p = 1.15 + 0.05/n*i                                 # Mean values of stock returns
sig = 0.05/450 * (2*i*n*(n+1))**0.5                 # Maximum deviation of stock returns
Gamma = 3                                           # Budget of uncertainty

model = ro.Model()                                  # Create a robust optimization model
x = model.dvar(n)                                   # Define decision variables x as an array
z = model.rvar(n)                                   # Define random variables z as an array

model.maxmin((p + sig*z) @ x,                       # Define the max-min objective function
             abs(z)<=1, at.norm(z, 1) <= Gamma)     # Uncertainty set of the model
model.st(sum(x) <= 1)                               # Define constraints
model.st(x >= 0)

model.solve(grb)                                    # Solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0029s


The solution as the stock allocation is given below.


```python
plt.plot(i, x.get())
plt.xlabel('stocks')
plt.ylabel('x')
plt.show()
```


<img src='https://github.com/XiongPengNUS/pyatom/blob/master/output_15_0.png?raw=true' width=500pt>


### A Knapsack problem: robust optimization v.s. robustness optimization

In this example, we use the <code>PyAtom</code> package to implement the robust and robustness optimization models described in the paper [The Dao of Robustness](http://www.optimization-online.org/DB_FILE/2019/11/7456.pdf).


```python
import pyatom.ro as ro
import pyatom.grb_solver as grb
import pyatom as at
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
    model.st(((w_hat + z*delta) @ x <= b).forall(abs(z) <= 1, at.norm(z, 1) <= r))

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


<img src='https://github.com/XiongPengNUS/pyatom/blob/master/output_27_0.png?raw=true' width=500pt>



<img src='https://github.com/XiongPengNUS/pyatom/blob/master/output_27_1.png?raw=true' width=500pt>


### Robust optimization with linear decision rules


```python
from pyatom import ro
import pyatom.grb_solver as grb
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, linewidth=200)
```

### Robust optimization for a lot-sizing problem

The robust model for the lot-sizing problem described in the paper [Duality in Two-Stage Adaptive Linear Optimization: Faster Computation and Stronger Bounds](https://www.fransderuiter.com/papers/BertsimasdeRuiter2016.pdf) is implemented with <code>PyAtom</code> to demonstrate how the linear decision rules is specified for adaptive decisions.

The parameters for a N=30 case are defined below.


```python
from pyatom import ro
import pyatom.grb_solver as grb
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
    Running time: 0.4217s


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


<img src='https://github.com/XiongPengNUS/test/blob/master/lot-sizing%20results.png?raw=true' width=500pt>



### Robust optimization for inventory

The AARC inventory model in the paper [Adjustable robust solutions of uncertain linear programs](https://www2.isye.gatech.edu/~nemirovs/MP_Elana_2004.pdf) is used here to demonstrate the implementation of linear decision rules for multi-stage problem.

```python
from pyatom import ro
import numpy as np
import pyatom as at
import pyatom.grb_solver as grb

T, I =24, 3                              
P, Q = 567, 13600
V0, Vmin, Vmax = 1500, 500, 2000
delta=0.2

t = np.arange(T).reshape((1, T))
d0 = 1000 * (1 + 1/2*np.sin(np.pi*(t/12)))
c =np.array([[1], [1.5], [2]]) @ (1 + 1/2*np.sin(np.pi*t/12))

model = ro.Model()

d = model.rvar(T)                       # Random demand
uset = (d <= (1+delta) * d0,
        d >= (1-delta) * d0)            # Uncertainty set of random demand

p = model.ldr((I, T))                   # decision rule for adaptive decision
for i in range(1, T):
    p[:, i].adapt(d[:i])                # Define LDR dependency

model.minmax((c*p).sum(), uset)         # The min-max objective function
model.st(p >= 0);                       # Constraints
model.st(p <= P);
model.st(p.sum(axis=1) <= Q)

v = V0
for i in range(T):
    v += p[:,i].sum() - d[i]
    model.st(v <= Vmax)
    model.st(v >= Vmin)

model.solve(grb)
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 2.0997s

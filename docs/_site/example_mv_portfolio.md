<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Mean-variance portfolio optimization

In this example, we consider a portfolio construction problem discussed in [Robust Solutions of Uncertain Linear Programs](https://www2.isye.gatech.edu/~nemirovs/stablpn.pdf). Suppose there are \\(n=150\\) stocks, and each stock \\(i\\) has the mean return to be \\(p_i\\) and the standard deviation to be \\(\sigma_i\\). Let \\(x_i\\) be the fraction of wealth invested in stock \\(i\\), a classic approach is to formulate the problem as a quadratic program, where a mean-variance objective function is maximized:

$$
\begin{align}
\max~&\sum\limits_{i=1}^np_ix_i - \phi \sum\limits_{i=1}^n \sigma_i^2x_i^2 \\
\text{s.t.}~&\sum\limits_{i=1}^nx_i = 1 \\
& x_i \geq 0, ~\forall i = 1, 2, ..., n,
\end{align}
$$

with the constant \\(\phi=5\\), and the means and standard deviations are specified to be

$$
\begin{align}
&p_i = 1.15 + i\frac{0.05}{150} & \forall i = 1, 2, ..., n\\
&\sigma_i = \frac{0.05}{450}\sqrt{2in(n+1)} & \forall i = 1, 2, ..., n.
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

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Box with the Maximum Volume

Here, we attempt to optimize the shape of a box, in terms of its width \\(w\\), height \\(h\\), and depth \\(d\\), so that its volume is maximized, subject to a number of constraints. The model can be written as

$$
\begin{align}
\max~& whd \\
\text{s.t.}~&2(wh + dh) \leq A_{\text{wall}} \\
&wd \leq A_{\text{floor}} \\
&\alpha \leq h/w \leq \beta, \gamma \leq d/w \leq \delta, 
\end{align}
$$

where model parameters:
- Limits on wall area \\(A_{\text{wall}} = 200\\),
- Limits on floor area \\(A_{\text{floor}} = 150\\),
- Limits on height-width raito: \\(\alpha=0.8\\), and \\(\beta=1.5\\),
- Limits on depth-width raito: \\(\gamma=0.8\\), and \\(\delta=1.5\\).

The geometric program above is reformulated as decision variables are replaced by their logarithm transformations: \\(x = \log(w)\\), \\(y = \log(h)\\), and \\(z = \log(d)\\), then we have

$$
\begin{align}
\max~& x + y + z \\
\text{s.t.}~&2(\exp(x + y) + \exp(z + y)) \leq A_{\text{wall}} \\
&\exp(x + z) \leq A_{\text{floor}} \\
&\exp(x - y) \leq \alpha, \exp(y - x) \leq \beta \\
&\exp(x - z) \leq \gamma, \exp(z - x) \leq \delta. \\
\end{align}
$$

Such an exponential cone program can be implemented by the code below.

```python
from rsome.rsome import ro
from rsome.rsome import eco_solver as eco
import rsome.rsome as rso
import numpy as np

A_wall = 200
A_floor = 150
alpha, beta = 0.8, 1.5
gamma, delta = 0.8, 1.5

model = ro.Model()
x = model.dvar()
y = model.dvar()
z = model.dvar()
a = model.dvar()
b = model.dvar()

model.max(x + y + z)
model.st(rso.exp(x + y) <= a)
model.st(rso.exp(z + y) <= b)
model.st(2 * (a+b) <= A_wall)
model.st(rso.exp(x + z) <= A_floor)
model.st(rso.exp(x - y) <= alpha, rso.exp(y - x) <= beta)
model.st(rso.exp(x - z) <= gamma, rso.exp(z - x) <= delta)
model.solve(eco)
```


    Being solved by ECOS...
    Solution status: Optimal solution found
    Running time: 0.0011s

The optimal solution is presented in the following code segment.

```python
x.get().round(4), y.get().round(4), z.get().round(4)
```
    (1.7329, 1.956, 2.1383)

Hence the optimal width, height, and depth are 5.6569, 7.0711, and 8.4853, respectively.

Please note that RSOME does not support the summation of two or more exponential functions, such as the exponential function `rso.exp()`. This is why we introduced intermediate variables `a` and `b` to represent the epigraph of `rso.exp(x + y)` and `rso.exp(z + y)`, so that their summation can be formulated in constraints.

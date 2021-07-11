<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Multi-Stage Stochastic Financial Planning

In this example we consider the multi-stage financial planning problem discussed in page 23 of [Birge and Francois (2011)](#ref1). As a multi-stage model problem, the here-and-now decision allocates the total wealth \\(d\\) between two investment types: stocks (S) and bonds (B). Each investment type may have a high return and a low return as two possible outcomes. It is assumed that the high returns for stocks and bonds are 1.25 and 1.14, and the low returns are 1.06 and 1.12, respectively. These outcomes are independent and with equal likelihood, so throughout the subsequent stages, we would have eight scenarios with equal probabilities. The random return outcomes of stocks and bonds are represented by a scenario tree shown below.

<img src="https://www.researchgate.net/profile/Zhi-Chen-21/publication/339817145/figure/fig4/AS:867492100591619@1583837642911/Scenario-tree-of-the-financial-planning-problem_W640.jpg" width=400>

Besides the deterministic equivalent of the stochastic model, the financial planning problem can also be formulated as a distributionally robust optimization problem where the decision tree is represented by the following ambiguity set,

$$
\mathcal{F} = \left\{
\mathbb{P}\in\mathcal{P}_0(\mathbb{R}^{3\times2}\times[S]) \left|
\begin{array}
~&(\tilde{\pmb{z}}, \tilde{s}) \in \mathbb{P} &&\\
&\mathbb{P}[\tilde{\pmb{z}} \in \mathcal{Z}_{\tilde{s}}|\tilde{s}=s]=1, &&\forall s\in[S]\\
&\mathbb{P}[\tilde{s}=s] = 1/S
\end{array}
\right.
\right\},
$$

with the scenario number \\(S=8\\) and the support \\(\mathcal{Z}_s=\\left\\{\hat{\pmb{z}}_s \\right\\} \\). The financial planning problem is thus written as

$$
\begin{align}
\max~&\inf \mathbb{E}\left[\overline{x}(\tilde{s}) - 4\underline{x}(\tilde{s})\right] \\
\text{s.t.}~&w_1, w_2 \geq 0, ~w_1 + w_2 = d  &&\\
&x_{11}(s) + x_{12}(s) - \pmb{z}_1^{\top}\pmb{w} = 0 && \forall \pmb{z} \in \mathcal{Z}_s, s\in [S]\\
&x_{21}(s) + x_{22}(s) - \pmb{z}_2^{\top}\pmb{x}_1(s) = 0 && \forall \pmb{z} \in \mathcal{Z}_s, s\in [S]\\
&\pmb{z}_3^{\top}\pmb{x}_2(s) + \overline{x}(s) - \underline{x}(s) = \tau \\
&\pmb{x}_1(s), \pmb{x}_2(s) \geq \pmb{0}, ~ \overline{x}(s), \underline{x}(s) \geq 0.
\end{align}
$$

According to the scenario tree, the adaptive decisions are governed by the event-wise static adaptation rules below,

$$
\begin{align}
&\pmb{x}_1(s) \in \mathcal{A}(\{1, 2, 3, 4\}, \{5, 6, 7, 8\}) \\
&\pmb{x}_2(s) \in \mathcal{A}(\{1, 2\}, \{3, 4\}, \{5, 6\}, \{7, 8\}) \\
&\overline{x}(s), \underline{x}(s) \in \mathcal{A}(\{1\}, \{2\}, \{3\}, \{4\}, \{5\}, \{6\}, \{7\}, \{8\}).
\end{align}
$$

In this example, the target parameter \\(\tau=80\\), and the initial budget is given as \\(d=55\\). These parameters and the scenario tree are defined by the following code cell.

```python
import numpy as np

tau = 80
d = 55

z_hat = np.array([[[0.25, 0.14], [0.25, 0.14], [0.25, 0.14]],
                  [[0.25, 0.14], [0.25, 0.14], [0.06, 0.12]],
                  [[0.25, 0.14], [0.06, 0.12], [0.25, 0.14]],
                  [[0.25, 0.14], [0.06, 0.12], [0.06, 0.12]],
                  [[0.06, 0.12], [0.25, 0.14], [0.25, 0.14]],
                  [[0.06, 0.12], [0.25, 0.14], [0.06, 0.12]],
                  [[0.06, 0.12], [0.06, 0.12], [0.25, 0.14]],
                  [[0.06, 0.12], [0.06, 0.12], [0.06, 0.12]]])
```

The Python code for implementing such a model is given as follows.

```python
from rsome import dro
from rsome import E
from rsome import grb_solver as grb


S = 8
model = dro.Model(S)                    # a model with S scenarios

z = model.rvar((3, 2))                  # random variables as a 3x2 array
fset = model.ambiguity()                # create an ambiguity set
for s in range(S):
    fset[s].suppset(z == 1 + z_hat[s])  # scenario-wise support of z
pr = model.p
fset.probset(pr == 1/S)                 # probability of each scenario

w = model.dvar(2)                       # 1st-stage decision w
x1 = model.dvar(2)                      # 2nd-stage decision x1
x2 = model.dvar(2)                      # 3rd-stage decision x2
x_over = model.dvar()                   # 4th-stage decision \overline{x}
x_under = model.dvar()                  # 4th-stage decision \underline{x}

for s in range(S):
    if s%4 == 0:
        x1.adapt(range(s, s+4))         # recourse adaptation of x1
    if s%2 == 0:
        x2.adapt(range(s, s+2))         # recourse adaptation of x2

    x_over.adapt(s)                     # recourse adaptation of \overline{x}
    x_under.adapt(s)                    # recourse adaptation of \underline{x}

# the objective and constraints
model.maxinf(E(x_over - 4*x_under), fset)
model.st(w >= 0, w.sum() == d)
model.st(x1.sum() == z[0]@w)
model.st(x2.sum() == z[1]@x1)
model.st(z[2]@x2 - x_over + x_under == tau)
model.st(x1 >= 0, x2 >= 0,
         x_over >= 0, x_under >= 0)

model.solve(grb)                        # solve the model by Gurobi
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0020s


The objective value is \\(-1.514\\), as retrieved by the `model.get()` command. The solution to the here-and-now decision \\(\pmb{w}\\) is given below.

```python
w.get().round(3)                        # solution of w
```


    array([41.479, 13.521])


In the following code segment, `x1.get()` returns the scenario-wise solution of \\(\pmb{x}_1\\) as a `pandas.Series` type object. Index labels of the series are the same as the scenario indices, and the solution as an array for each scenario is an element of the series, which could be accessed by the `loc` or `iloc` indexers.

```python
x1_sol = x1.get()
x1_sol.apply(lambda x: x.round(3))      # scenario-wise solution of x1
```

    0     [65.095, 2.168]
    1     [65.095, 2.168]
    2     [65.095, 2.168]
    3     [65.095, 2.168]
    4    [36.743, 22.368]
    5    [36.743, 22.368]
    6    [36.743, 22.368]
    7    [36.743, 22.368]
    dtype: object


Similarly, scenario-wise solutions of \\(\pmb{x}_2\\), \\(\overline{x}\\), and \\(\underline{x}\\) are presented in the similar format.

```python
x2_sol = x2.get()
x2_sol.apply(lambda x: x.round(3))      # scenario-wise solution of x2
```


    0     [83.84, 0.0]
    1     [83.84, 0.0]
    2    [0.0, 71.429]
    3    [0.0, 71.429]
    4    [0.0, 71.429]
    5    [0.0, 71.429]
    6      [64.0, 0.0]
    7      [64.0, 0.0]
    dtype: object


```python
x_over_sol = x_over.get()
x_over_sol.apply(lambda x: x.round(3))  # scenario-wise solution of x_over
```

    0     [24.8]
    1     [8.87]
    2    [1.429]
    3      [0.0]
    4    [1.429]
    5      [0.0]
    6      [0.0]
    7      [0.0]
    dtype: object


```python
x_under_sol = x_under.get()
x_under_sol.apply(lambda x: x.round(3)) # scenario-wise solution of x_under
```

    0      [0.0]
    1      [0.0]
    2      [0.0]
    3      [0.0]
    4      [0.0]
    5      [0.0]
    6      [0.0]
    7    [12.16]
    dtype: object

<br>
#### Reference

<a id="ref1"></a>

Birge, John R, Francois Louveaux. 2011. [<i>Introduction to stochastic programming</i>](https://www.springer.com/gp/book/9781461402367). Springer.

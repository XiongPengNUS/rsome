<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Robust Vehicle Pre-Allocation

In this example,  we consider the vehicle pre-allocation problem introduced in [Hao et al. (2020)](#ref2). Suppose that there are \\(I\\) supply nodes and \\(J\\) demand nodes in an urban area.  The operator,  before the random demand \\(\tilde{d}\_j = (\tilde{d})\_{j\in[J]}\\) realizes, allocates \\(x_{ij}\\) vehicles from supply node \\(i\in[I]\\) (which has a numbers \\(i\\) of idle vehicles) to demand node \\(j\in[J]\\) at a unit cost \\(c_{ij}\\), and the revenue is calculated as \\(\sum_{j \in [J]} r_j \min\\left\\{\sum_{i \in [I]} x_{ij}, d_j\\right\\}\\), as the uncertain demand is realized. Following the work done by [Hao et al. (2020)](#ref2), model parameters are summarized as follows:
- Number of supply nodes \\(I=1\\);
- Number of demand nodes \\(J=10\\);
- Revenue coefficients \\(\pmb{r}=(4.50, 4.41, 3.61, 4.49, 4.38, 4.58, 4.53, 4.64, 4.58, 4.32)\\);
- Cost coefficients \\(c_j = 3\\), where \\(j=1, 2, ..., J\\);
- Maximum supply of vehicles \\(q_i=400\\), where \\(i=1, 2, ..., I\\).

```python
import numpy as np

I, J = 1, 10
r = np.array([4.50, 4.41, 3.61, 4.49, 4.38, 4.58, 4.53, 4.64, 4.58, 4.32])
c = 3 * np.ones((I, J))
q = 400 * np.ones(I)
```

The vehicle pre-allocation will be solved by the robust and sample robust optimization (proposed by [Bertsimas et al. 2021](#ref1)) approaches using the RSOME `ro` framework.

#### The Robust Model

The vehicle pre-allocation decision under demand uncertainty can be made by solving the robust optimization problem below:

$$
\begin{align}
\min\limits_{\pmb{x}, \pmb{y}}~&\max\limits_{\pmb{d}\in\mathcal{Z}}\left\{\sum\limits_{i\in[I]}\sum\limits_{j\in[J]}(c_{ij} - r_j)x_{ij} + \sum\limits_{j\in[J]}r_jy_j(\pmb{d})\right\} \hspace{-1.5in}&& \\
\text{s.t.}~&y_j(\pmb{d}) \geq \sum\limits_{i\in[I]}x_{ij} - d_j && \forall \pmb{d} \in \mathcal{Z}, \forall j \in [J] \\
&y_j(\pmb{d}) \geq 0 && \forall \pmb{d} \in \mathcal{Z}, \forall j \in [J] \\
&y_j\in\mathcal{L}([J]) && \forall j \in [J] \\
&\sum\limits_{j\in[J]}x_{ij} \leq q_i && \forall i \in [I] \\
&x_{ij} \geq 0 &&\forall i \in[I], \forall j \in [J], \\
\end{align}
$$

where the wait-and-see decision \\(\pmb{y}\\) that represents the bookkeeping revenue is approximated by a linear decision rule \\(\mathcal{L}([J])\\), implying that each \\(y_j\\) affinely depends on the demand realization \\(\pmb{d}\\). Here \\(\mathcal{Z}\\) is a box uncertainty set where the upper and lower bounds are identified using the sample demand dataset [taxi_rain.csv](taxi_rain.csv).

```python
import pandas as pd

data = pd.read_csv('https://xiongpengnus.github.io/rsome/taxi_rain.csv')

demand = data.loc[:, 'Region1':'Region10']      # taxi demand data

d_ub = demand.max().values                      # upper bound of demand
d_lb = demand.min().values                      # lower bound of demand
```

Then the robust model can be implemented by the following code segment.

```python
from rsome import ro                            # import the ro module
from rsome import grb_solver as grb             # import Gurobi solver interface

model = ro.Model()                              # create an RO model

d = model.rvar(J)                               # create an array of random demand
zset = (d <= d_ub, d >= d_lb)                   # define a box uncertainty set

x = model.dvar((I, J))                          # define here-and-now decision x
y = model.ldr(J)                                # define linear decision rule y
y.adapt(d)                                      # y affinely adapts to d

model.minmax(((c-r)*x).sum() + r@y, zset)       # the worst-case objective function
model.st(y >= x.sum(axis=0) - d, y >= 0)        # robust constraints
model.st(x.sum(axis=1) <= q, x >= 0)            # deterministic constraints

model.solve(grb)                                # solve the model with Gurobi
```

```
Being solved by Gurobi...
Solution status: 2
Running time: 0.0016s
```

The optimal vehicle pre-allocation decision is \\(\pmb{x}=\\)(0, 0, 0, 0, 0, 39.6138, 0, 0, 0, 0), which is rather conservative, and the optimal objective value is \\(-62.59\\).

#### The Sample Robust Model

[Bertsimas et al. (2021)](#ref1) recently proposed a two-stage sample robust model where a collection \\(\left\\{\hat{\pmb{d}}_1, \hat{\pmb{d}}, \dots \hat{\pmb{d}}_S\right\\}\\) of historical demand samples are integrated into the decision-making process. In the context of vehicle pre-allocation, the sample robust model can be written as the following two-stage problem:

$$
\begin{align}
\min\limits_{\pmb{x}, \pmb{y}}~&\sum\limits_{i\in[I]}\sum\limits_{j\in[J]}(c_{ij} - r_j)x_{ij} + \frac{1}{S}\sum\limits_{s\in[S]}a_s \hspace{-1.5in}&& \\
\text{s.t.}~&a_s \geq \sum\limits_{j\in[J]}r_jy_{sj}(\pmb{d}) &&\forall \pmb{d} \in \mathcal{Z}_s, s \in [S] \\
&y_{sj}(\pmb{d}) \geq \sum\limits_{i\in[I]}x_{ij} - d_j && \forall \pmb{d} \in \mathcal{Z}_s, j \in [J], s \in [S] \\
&y_{sj}(\pmb{d}) \geq 0 && \forall \pmb{d} \in \mathcal{Z}_s, j \in [J], s \in [S] \\
&y_{sj}\in\mathcal{L}([J]) && \forall j \in [J], s \in [S] \\
&\sum\limits_{j\in[J]}x_{ij} \leq q_i && \forall i \in [I] \\
&x_{ij} \geq 0 &&\forall i \in[I], \forall j \in [J], \\
\end{align}
$$


where \\(\pmb{a}\in\mathbb{R}^S\\) is a vector of intermediate variables representing the worst-case recourse cost in each scenario, and \\(\mathcal{Z}_s=\\left\\{\pmb{d} \in \left[\underline{\pmb{d}}, \bar{\pmb{d}}\right] \left\vert \left\\|\pmb{d} - \hat{\pmb{d}}_s \right\\| \leq \varepsilon \right. \\right\\}\\) is an \\(\varepsilon\\)-neighborhood uncertainty set defined by a general norm \\(\\|\cdot\\|\\) around each demand sample \\(\hat{\pmb{d}}_s\\). Note that the multiple-policy approximation of the wait-and-see decision \\(\pmb{y}\\) allows different affine mappings for each demand sample \\(\hat{\pmb{d}}_s\\), thus the two-dimensional decision rule \\(\left(\pmb{y}(\pmb{d})\right)\_{s\in[S], j\in[J]}\\). Such a sample robust model (assuming the conservatism parameter \\(\varepsilon=0.25\\)) is implemented by the following code segment.

```python
from rsome import ro                                    # import the ro module
from rsome import norm                                  # import the norm function
from rsome import grb_solver as grb                     # import the Gurobi interface

dhat = demand.values                                    # sample demand as an array
S = dhat.shape[0]                                       # sample size of the dataset
epsilon = 0.25                                          # parameter of robustness

model = ro.Model()                                      # create an RO model

d = model.rvar(J)                                       # random variable d
a = model.dvar(S)                                       # variable as the recourse cost
x = model.dvar((I, J))                                  # here-and-now decision x
y = model.ldr((S, J))                                   # linear decision rule y
y.adapt(d)                                              # y affinely adapts to d

model.min(((c-r)*x).sum() + (1/S)*a.sum())              # minimize the objective
for s in range(S):
    zset = (d <= d_ub, d >= d_lb,
            norm(d - dhat[s]) <= epsilon)               # uncertainty set for the sth sample
    model.st((a[s] >= r@y[s]).forall(zset))             # constraints for the sth sample
    model.st((y[s] >= x.sum(axis=0) - d).forall(zset))  # constraints for the sth sample
    model.st((y[s] >= 0).forall(zset))                  # constraints for the sth sample
model.st(x.sum(axis=1) <= q, x >= 0)                    # constraints

model.solve(grb)                                        # solve the model by Gruobi
```

The optimal vehicle pre-allocation decision is \\(\pmb{x}=\\)(0.341, 0.358, 0,  4.289, 0, 69.456, 2.452, 4.578, 5.229, 2.486), and the optimal objective value is \\(-103.656\\). Notice that in a special case where \\(\varepsilon=0\\), the sample robust model is equivalent to the sample average approximation approach.

We would like to highlight that the `ro` framework enables users to specify  different uncertainty sets for the objective function and each of the constraints: in the sample robust model above, different uncertainty sets are defined around samples and these sets for constraints can be easily specified by calling the `forall()` method. Besides using the `ro` framework, the `dro` module also provides neat and highly readable tools for implementing the sample robust model. We refer interested readers to the page [Distributionally Robust Vehicle Pre-Allocation](example_dro_vehicle).


<br>

#### Reference

<a id="ref1"></a>

Bertsimas, Dimitris, Shimrit Shtern, and Bradley Sturt. 2021. [Two-stage sample robust optimization](https://pubsonline.informs.org/doi/abs/10.1287/opre.2020.2096). <i>Operations Research</i>.

<a id="ref2"></a>

Hao, Zhaowei, Long He, Zhenyu Hu, and Jun Jiang. 2020. [Robust vehicle pre‐allocation with uncertain covariates]((https://onlinelibrary.wiley.com/doi/abs/10.1111/poms.13143)). <i>Production and Operations Management</i> <b>29</b>(4) 955-972.

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Multi-Stage Inventory Control

In this example we consider a multi-stage inventory control problem to showcase how different distributionally
robust models can be deployed in RSOME with little switching cost.  The inventory control problem assumes that at the beginning of each time period \\(t\in[T]\\), the decision maker starts with \\(I_{0}\\) units of product in inventory and selects a production quantity \\(x_t \in [0, \bar{x}_t]\\) with zero lead time at a unit cost \\(c_t\\). After realization of demand \\(d_t \geq 0\\), the inventory is updated to \\(I_t = I\_{t−1}+x_t −d_t\\) and it may incur a holding cost of \\(h_t \max\\{I_t, 0\\}\\) or a backorder cost of \\(b_t \max\\{−I_t, 0\\}\\). We use the same parameters and data generation as in [See and Sim (2010)](#ref4). Specifically, the demand follows a non-stationary autoregressive stochastic process of the form:

$$
\tilde{d}_t(\pmb{v}) = \tilde{v}_t + \alpha \tilde{v}_{t-1} + \alpha \tilde{v}_{t-2} + \cdots + \alpha \tilde{v}_1 + \mu = d_{t-1}(\tilde{\pmb{v}}) - (1-\alpha)\tilde{v}_{t-1} + \tilde{v}_t ~~~\forall t \in [T],
$$

where \\(\tilde{v}_1,\dots,\tilde{v}_T\\) are independent and identically distributed uniform random variables in \\([-\bar{v}, \bar{v}]\\), and other parameters are given as:
- Number of time steps: \\(T=10\\).
- Constant in the autoregressive process: \\(\alpha=0.25\\).
- Initial inventory status: \\(I_0=0\\).
- Bounds of each random variable: \\(\bar{v}=10\\).
- Mean demand: \\(\mu_t=200\\).
- Upper bound of the order quantity: \\(\bar{x}_t=260\\).
- Order cost: \\(c_t=0.1\\).
- Holding cost: \\(h_t=0.02\\).
- Backorder cost: \\(b_t=0.4\\) if \\(t=1, 2, \cdots, T-1\\), and \\(b_T=4.0\\).

These parameters are defined by the code segment below.

```python
import numpy as np

T = 10		
alpha = 0.25
I0 = 0
vbar = 10
mu = 200 * np.ones(T)
xbar = 260 * np.ones(T)
c = 0.1 * np.ones(T)
h = 0.02 * np.ones(T)
b = 0.4 * np.ones(T)
b[T-1] = 4.0
```

With a null inventory on hand (<i>i.e</i>, \\(I_0= 0\\)), the decision maker dynamically selects production quantities to minimize the expected total cost over the entire planning horizon, given by

$$
\begin{align}
\min_{\pmb{x}, ~\pmb{y}} ~& \sup_{\mathbb{P} \in \mathcal{F}}\mathbb{E}_{\mathbb{P}}\left[c_1 x_1 + \sum_{t\in[T]\setminus\{1\}}c_t x_t(\tilde{s}, \tilde{\pmb{z}}) + \sum_{t\in[T]}y_t(\tilde{s}, \tilde{\pmb{z}})\right] \hspace{-0.4in} &\\
\text{s.t.}~ & I_1(s, \pmb{z}) = I_0 + x_1 - d_{1}(s, \pmb{z}) & \forall \pmb{z} \in \mathcal{Z}_s, \; s \in [S] \\
&I_t(s, \pmb{z}) = I_{t-1}(s, \pmb{z}) + x_t(s, \pmb{z}) - d_t(s, \pmb{z}) & \forall \pmb{z} \in \mathcal{Z}_s, \; s \in [S], \; t \in [T]\setminus\{1\} \\
&y_t(s, \pmb{z}) \geq h_t I_t(s, \pmb{z}) & \forall \pmb{z} \in \mathcal{Z}_s, \; s \in [S], \; t \in [T] \\
&y_t(s, \pmb{z}) \geq -b_t I_t(s, \pmb{z}) & \forall \pmb{z} \in \mathcal{Z}_s, \; s \in [S], \; t \in [T] \\
& 0 \leq x_1 \leq \bar{x}_1 \\
& 0 \leq x_t(s, \pmb{z}) \leq \bar{x}_t & \forall \pmb{z} \in \mathcal{Z}_s, \; s\in [S], \; t \in [T]\setminus\{1\} \\
& x_t \in \overline{\mathcal{A}}(\mathcal{C}, \mathcal{J}_{t-1}) &\forall t \in [T]\setminus\{1\} \\
& y_t, \; I_t \in \overline{\mathcal{A}}(\mathcal{C}, \mathcal{J}_t), &\forall t \in [T].
\end{align}
$$

With different choices of \\(\mathcal{F}\\) based on a number \\(S\\) of historical demand samples \\(\hat{\pmb{d}}_1,\cdots,\hat{\pmb{d}}_S\\) (<i>i.e.</i>, \\(\hat{\pmb{v}}_1,\cdots,\hat{\pmb{v}}_S\\)), the array of random variables \\(\tilde{\pmb{z}}\\) (including those auxiliary ones) may be formulated differently.  We next present some examples of approaches proposed in the literature—all fit well in the distributionally robust optimization framework of RSOME

#### Partial Cross-Moment
Since random factors \\(\tilde{\pmb{v}}\_1,\cdots,\tilde{\pmb{v}}\_S\\) are uncorrelated and they share an identical standard deviation \\(\bar{v}/\sqrt{3}\\), the standard deviation of a partial \\(\sum_{\tau=r}^t\tilde{v}_{\tau}\\) can be bounded by \\(\bar{v}\sqrt{(t-r+1)/3}\\).  Let \\(\tilde{\pmb{z}}=(\tilde{\pmb{v}}, \tilde{\pmb{u}})\\), where \\(\tilde{\pmb{u}}\in\mathbb{R}^{T(T+1)/2}\\) is a vector of auxiliary random variables that captures the partial cross-moment  information, the ambiguity set \\(\mathcal{F}\\) can then be formulated as follows

$$
\mathcal{F} = \left\{\mathbb{P} \in \mathcal{P}_0 (\mathbb{R}^T \times \mathbb{R}^{T(T+1)/2} \times \{1\}) ~\left|~
\begin{array}{l@{\quad}l}
((\tilde{\pmb{v}},\tilde{\pmb{u}}), \tilde{s}) \sim \mathbb{P} \\
\mathbb{E}_{\mathbb{P}}[\tilde{\pmb{v}} \mid \tilde{s} = 1] = \pmb{0} \\
\mathbb{E}_{\mathbb{P}}[\tilde{u}_{rt} \mid \tilde{s} = 1] \leq \phi_{rt} & \forall r \leq t, \; t \in [T] \\
\mathbb{P}[(\pmb{v},\tilde{\pmb{u}}) \in \mathcal{Z}_s \mid \tilde{s} = 1] = 1
\end{array}
\right.
\right\},
$$

where the variance \\(\phi_{rt}=(\bar{v}^2(t-r+1))/3\\) and the support set takes the form

$$
\mathcal{Z}_s = \left\{(\pmb{v}, \pmb{u}) \in \mathbb{R}^T \times \mathbb{R}^{T(T+1)/2}: \;
\pmb{v} \in [\underline{\pmb{v}}, \bar{\pmb{v}}], \; u_{st} \geq \left(\sum_{\tau=r}^t v_{\tau}\right)^2 ~~~\forall s \leq t, \; t \in [T]
\right\},
$$

and the corresponding event-wise recourse adaptation is characterized by

$$
\begin{cases}
x_t \in \overline{\mathcal{A}}(\{\{1\}\}, \mathcal{J}_{t-1}) &~\forall t \in [T]\setminus\{1\} \\
y_t, \; I_t \in \overline{\mathcal{A}}(\{\{1\}\}, \mathcal{J}_t) &~\forall t \in [T],
\end{cases}
$$

where for each \\(t\in[T]\\), \\(\mathcal{J}_t\\) consists of (i) the indices of \\(\tilde{v}_1\\) to \\(\tilde{v}_t\\) and (ii) the indices of \\(\tilde{u}\_{\tau r}\\) such that \\(\tau \leq r \leq t\\). The model can be formulated and solved by the code below.

```python
from rsome import dro				
from rsome import E
from rsome import square
from rsome import grb_solver as grb
import numpy as np

phi = np.array([vbar*((t-r+1)/3)
               for t in range(T)
               for r in range(t+1)]) # ambiguity set parameter phi

model = dro.Model()

v = model.rvar(T)                    # random variable v
u = model.rvar(T*(T+1)//2)           # auxiliary random variable u
fset = model.ambiguity()             # create an ambiguity set
fset.exptset(E(v) == 0, E(u) <= phi) # define the uncertainty set of expectations
fset.suppset(v >= -vbar, v <= vbar,
             [square(v[r:t].sum()) <= u[t*(t-1)//2 + r]
              for t in range(T+1)
              for r in range(t)])    # define the support of random variables
temp = alpha*np.triu(np.ones(T), 0) + (1-alpha)*np.eye(T)
d = v@temp + mu                      # random demand

x1 = model.dvar()                    # variable x1
x = model.dvar(T-1)                  # an array of variables x2, x3, ..., xT
I = model.dvar(T)                    # an array of variables I1, I2, ..., IT
y = model.dvar(T)                    # an array of variables y1, y2, ..., yT
for t in range(T-1):
    x[t].adapt(v[:t+1])              # x[t] adapts to v[0], ..., v[t]
    x[t].adapt(u[:(t+1)*(t+2)//2])   # x[t] adapts to u[0], ..., u[t*(t+3)//2]
for t in range(T):
    I[t].adapt(v[:t+1])              # I[t] adapts to v[0], ..., v[t]
    I[t].adapt(u[:(t+1)*(t+2)//2])   # I[t] adapts to u[0], ..., u[t*(t+3)//2]
    y[t].adapt(v[:t+1])              # y[t] adapts to v[0], ..., v[t]
    y[t].adapt(u[:(t+1)*(t+2)//2])   # y[t] adapts to u[0], ..., u[t*(t+3)//2]

model.minsup(E(c[0]*x1 + c[1:]@x + y.sum()), fset)
model.st(I[0] == I0 + x1 - d[0])
for t in range(1, T):
    model.st(I[t] == I[t-1] + x[t-1] - d[t])
model.st(y >= h*I, y >= -b*I)				
model.st(x1 >= 0, x1 <= xbar[0])				
model.st(x >= 0, x <= xbar[1:])

model.solve(grb)                     # solve the model by Gurobi
```

Note that when specifying the affine adaptation of decisions `x[t]`, `I[t]`, and `y[t]`, the indices are different from the mathematical formulation. For example, here `x[0]` and `v[0]` are respectively \\(x_2\\) and \\(v_1\\) in the mathematical formulation, so the index set \\(\mathcal{J}_{t-1}\\) is specified by the a loop such that for all `t in range(T-1)`, `x[t]` adapts to random variables `v[:t+1]` and `u[:(t+1)*(t+2)//2]`, as defined in each iteration.

#### Type-1 Wasserstein Metric

According to <b>Theorem 2</b> in [Chen et al. (2020)](#ref3), we consider a data-driven ambiguity set based on the type-1 Wasserstein metric, which is representable in the format of an event-wise ambiguity set

$$
\mathcal{F} = \left\{
\mathbb{P} \in \mathcal{P}_0 (\mathbb{R}^{T} \times \mathbb{R} \times [S])
~\left|~
\begin{array}{l@{\quad}l}
((\tilde{\pmb{v}}, \tilde{u}),\tilde{s}) \sim \mathbb{P} \\
\mathbb{E}_{\mathbb{P}}[\tilde{u} \mid \tilde{s} \in [S]] \leq \theta \\
\mathbb{P}[(\tilde{\pmb{z}},\tilde{u}) \in \mathcal{Z}_s \mid \tilde{s} = s] = 1 & \forall s \in [S] \\
\mathbb{P}[\tilde{s} = s] = \frac{1}{S} & \forall s \in [S]
\end{array}
\right.
\right\},
$$

where \\(S\\) is the sample size of the dataset, and for each sample \\(s\\), the support set is defined as \\(\mathcal{Z}_s = \left\\{(\pmb{v},u) \in \mathbb{R}^{T\times R}: \pmb{v}\in [\bar{\pmb{v}}, -\bar{\pmb{v}}], u \geq\|\pmb{v}−\hat{\pmb{v}}_s\| \right\\}\\). In this case, \\(\tilde{\pmb{z}} = (\tilde{\pmb{v}}, \tilde{u}) \\) with \\(\tilde{u}\\) being an auxiliary random scalar. The accompanying event-wise recourse adaptation is given by

$$
\begin{cases}
x_t \in \overline{\mathcal{A}}(\{\{1,\dots,S\}\}, \mathcal{J}_{t-1}) &~\forall t \in [T] \setminus \{1\} \\
y_t, \; I_t \in \overline{\mathcal{A}}(\{\{1,\dots,S\}\}, \mathcal{J}_t) &~\forall t \in [T],
\end{cases}
$$

where for each \\(t \in [T−1]\\), \\(\mathcal{J}_t\\) corresponds to the indices of \\(\tilde{v}_1\\) to \\(\tilde{v}_t\\); while \\(\mathcal{J}_T\\) contains the indices of \\(\tilde{v}_1\\) to \\(\tilde{v}_T\\) as well as that of \\(\tilde{u}\\).  In this example, the sample size of the demand data is \\(S=50\\) and the robustness parameter \\(\theta=0.02T\\), then the model can be implemented by the sample code below.

```python
from rsome import dro
from rsome import E
from rsome import norm
from rsome import grb_solver as grb
import numpy.random as rd

S = 50                              # sample size S=50
vhat = (1-2*rd.rand(S, T)) * vbar   # random sample of v
theta = 0.02 * T                    # Wasserstein metric parameter

model = dro.Model(S)                # a model with S scenarios

v = model.rvar(T)                   # random variable array v
u = model.rvar()                    # auxiliary variable u
fset = model.ambiguity()            # create an ambiguity set
fset.exptset(E(u) <= theta)         # define the uncertainty set for expectations
for s in range(S):                  # a loop to define the scenario-wise support
    fset[s].suppset(v >= -vbar, v <= vbar,
                    norm(v - vhat[s], 1) <= u)
pr = model.p
fset.probset(pr == 1/S)
temp = alpha*np.triu(np.ones(T), 0) + (1-alpha)*np.eye(T)
d = v@temp + mu                         

x1 = model.dvar()                   # variable x1
x = model.dvar(T-1)                 # an array of variables x2, x3, ..., xT
I = model.dvar(T)                   # an array of variables I1, I2, ..., IT
y = model.dvar(T)                   # an array of variables y1, y2, ..., yT
for t in range(T-1):
    x[t].adapt(v[:t+1])             # x[t] adapts to v[0], v[1], ..., v[t]
for t in range(T):
    I[t].adapt(v[:t+1])             # I[t] adapts to v[0], v[1], ..., v[t]
    y[t].adapt(v[:t+1])             # y[t] adapts to v[0], v[1], ..., v[t]
I[T-1].adapt(u)                     # I[T-1] adapt to all u
y[T-1].adapt(u)                     # y[T-1] adapt to all u

model.minsup(E(c[0]*x1 + c[1:]@x + y.sum()), fset)
model.st(I[0] == I0 + x1 - d[0])
for t in range(1, T):
    model.st(I[t] == I[t-1] + x[t-1] - d[t])
model.st(y >= h*I, y >= -b*I)
model.st(w >= 0, w <= xbar[0])
model.st(x >= 0, x <= xbar[1:])

model.solve(grb)                    # solve the model by Gurobi
```
Note that the ambiguity set is defined upon \\(S\\) scenarios, and the scenario number is specified when creating the RSOME model using the `dro.Model()` function. For each scenario `s`, the support set is defined by calling the `fset[s].suppset()` method in a for loop.  For all adaptive decisions in this example, we leave the scenario partitions unspecified since the default setting is \\(\mathcal{C}=\\{\\{1,2,\cdots,S\\}\\}\\), implying that adaptive decisions do not adapt to scenarios; see also [Bertsimas et al. (2019a)](#ref1).

#### Type-\\(\infty\\) Wasserstein Metric

The sample robust approach (which is based on the type-\\(\infty\\) Wasserstein metric) in [Bertsimas et al. (2019b)](#ref2) can be readily extended to this multi-stage problem. The type-\\(\infty\\) Wasserstein ambiguity set for the random variable \\(\tilde{\pmb{v}}\\) and random scenario \\(\tilde{s}\\) is given by

$$
\mathcal{F} = \left\{\mathbb{P} \in \mathcal{P}_0 (\mathbb{R}^T \times [S]) ~\left|~
\begin{array}{l@{\quad}l}			
(\tilde{\pmb{v}},\tilde{s}) \sim \mathbb{P} & \\			
\mathbb{P}[\pmb{v} \in [\underline{\pmb{v}}, \bar{\pmb{v}}], \; \|\tilde{\pmb{v}} - \hat{\pmb{v}}_s\| \leq \theta \mid \tilde{s} = s] = 1 & \forall s \in [S] \\
\mathbb{P}[\tilde{s} = s] = \frac{1}{S} & \forall s \in [S]
\end{array}
\right.
\right\},
$$

while the event-wise recourse adaptation is specified as

$$
\begin{cases}
x_t \in \overline{\mathcal{A}}(\{\{1\},\cdots,\{S\}\}, \mathcal{J}_{t-1}) &~\forall t \in [T] \setminus \{1\} \\[1mm]
y_t, \; I_t \in \overline{\mathcal{A}}(\{\{1\},\cdots,\{S\}\}, \mathcal{J}_t) &~\forall t \in [T].
\end{cases}
$$

Here, for each \\(t \in [T]\\), \\(\mathcal{J}_t\\) corresponds to the indices of \\(\tilde{v}_1\\) to \\(\tilde{v}_t\\) and the specific partition \\(\\{\\{1\\},\cdots,\\{S\\}\\}\\) implies that the adaptive decisions can adapt to each scenario. The type-\\(\infty\\) is implemented by the code segment below.

```python
from rsome import dro
from rsome import E
from rsome import norm
from rsome import grb_solver as grb
import numpy.random as rd

S = 50                              # sample size S=50
vhat = (1-2*rd.rand(S, T)) * vbar   # random sample of v
theta = 0.02 * T                    # maximum radius of the Wasserstein ball

model = dro.Model(S)                # a model with S scenarios

v = model.rvar(T)                   # # random variable array v
fset = model.ambiguity()            # create an ambiguity set
for s in range(S):                  # a loop to define the scenario-wise support
    fset[s].suppset(v >= -vbar, v <= vbar,
                    norm(v - vhat[s], 1) <= theta)
pr = model.p
fset.probset(pr == 1/S)
temp = alpha*np.triu(np.ones(T), 0) + (1-alpha)*np.eye(T)
d = v@temp + mu

x1 = model.dvar()                   # variable x1
x = model.dvar(T-1)                 # an array of variables x2, x3, ..., xT
I = model.dvar(T)                   # an array of variables I1, I2, ..., IT
y = model.dvar(T)                   # an array of variables y1, y2, ..., yT
for t in range(T-1):
    x[t].adapt(v[:t+1])             # x[t] adapts to v[0], v[1], ..., v[t]
for t in range(T):
    I[t].adapt(v[:t+1])             # I[t] adapts to v[0], v[1], ..., v[t]
    y[t].adapt(v[:t+1])             # y[t] adapt to v[0], v[1], ..., v[t]
for s in range(S):                  
    x.adapt(s)                      # x adapts to each scenario s
    y.adapt(s)                      # y adapts to each scenario s
    I.adapt(s)                      # I adapts to each scenario s

model.minsup(E(c[0]*x1 + c[1:]@x + y.sum()), fset)
model.st(I[0] == I0 + x1 - d[0])
for t in range(1, T):
    model.st(I[t] == I[t-1] + x[t-1] - d[t])
model.st(y >= h*I, y >= -b*I)
model.st(x >= 0, x <= xbar[1:])

model.solve(grb)                    # solve the model by Gurobi
```

In this example, the scenario partition \\(\mathcal{C}=\\{\\{1\\},\cdots,\\{S\\}\\}\\) is defined by a loop where in each iteration, the `adapt()` method is used to indicate that the given decision variable adapts to the scenario `s`.

After the model is successfully solved, we may further evaluate values of the scenario-wise decisions (and their expressions) under different realizations of random variables. For example, the expression `y(v.assign(vhat[0])` returns a series of arrays, and each array is the value of \\(\pmb{y}(s, \pmb{v})\\) in the \\(s\\)th scenario and \\(\pmb{v} = \hat{\pmb{v}}_1\\). In this function call of `y()`, the decision rules in different scenarios are evaluated under the same value of \\(\pmb{v}\\), which is the first historical record `vhat[0]`. We may also use the expression `y(v.assign(vhat, sw=True))` to enforce scenario-wise evaluation (keyword argument `sw=True`) of the decision rule, where the realizations of random variables are assumed to be `vhat[0]`, `vhat[1]`, ..., `vhat[S-1]`, respectively, for these \\(S\\) scenarios. 


#### Reference

<a id="ref1"></a>

Bertsimas, Dimitris, Christopher McCord, Bradley Sturt. 2019a. [Dynamic optimization with side information](https://arxiv.org/abs/1907.07307). <i>arXiv preprint arXiv:1907.07307</i>.

<a id="ref2"></a>

Bertsimas, Dimitris, Shimrit Shtern, and Bradley Sturt. 2021. [Two-stage sample robust optimization](https://pubsonline.informs.org/doi/abs/10.1287/opre.2020.2096). <i>Operations Research</i>.

<a id="ref3"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.

<a id="re43"></a>

See, Chuen-Teck, Melvyn Sim. 2010. [Robust approximation to multiperiod inventory management](https://pubsonline.informs.org/doi/abs/10.1287/opre.1090.0746). <i>Operations Research</i> <b>58</b>(3) 583–594.

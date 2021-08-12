<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Multi-Item Newsvendor Problem with Wasserstein Ambiguity Sets

In this example, we consider the multi-item newsvendor problem discussed in the paper [Chen et al. (2020)](#ref1). This newsvendor problem determines the order quantity \\(w_i\\) of each of the \\(I\\) products under a total budget \\(d\\). The unit selling price and ordering cost for each product item are denoted by \\(p_i\\) and \\(c_i\\), respectively. The uncertain demand of each product item is denoted by the random variable \\(\tilde{z}_i\\). Once the demand realizes, the selling quantity \\(y_i\\) is expressed as \\(\min\{x_i, z_i\}\\), and the newsvendor problem can be written as the following distributionally robust optimization model,

$$
\begin{align}
\min~& -\pmb{p}^{\top}\pmb{x} + \sup\limits_{\mathbb{P}\in\mathcal{F}(\theta)}\mathbb{E}_{\mathbb{P}}\left[\pmb{p}^{\top}\pmb{y}(\tilde{s}, \tilde{\pmb{z}}, \tilde{u})\right] && \\
\text{s.t.}~&\pmb{y}(s, \pmb{z}, u) \geq \pmb{x} - \pmb{z} && \forall (\pmb{z}, \pmb{u}) \in \mathcal{Z}_s, ~\forall s \in [S] \\
& \pmb{y}(s, \pmb{z}, u) \geq \pmb{0} && \forall (\pmb{z}, \pmb{u}) \in \mathcal{Z}_s, ~\forall s \in [S]\\
& y_i \in \overline{\mathcal{A}}(\{\{1\}, \{2\}, \dots, \{S\}\}, [I+1]) &&\forall i \in [I]\\
& \pmb{c}^{\top}\pmb{x} = d, ~ \pmb{x} \geq \pmb{0}
\end{align}
$$    

with \\(s\\) the scenario index, and \\(u\\) the auxiliary random variable. The recourse decision \\(y_i\\) follows the approximated adaptation \\(\overline{\mathcal{A}}(\\{\\{1\\}, \\{2\\}, \dots, \\{S\\}\\}, [I+1])\\) indicating that \\(y_i\\) adapts to different scenarios \\(s\\) and is affinely adaptive to the random variables \\(\pmb{z}\\) and the auxiliary variable \\(u\\). The model maximizes the worst-case expectation over a Wasserstein ambiguity set \\(\mathcal{F}\\), expressed as follows.

$$
\begin{align}
\mathcal{F}(\theta) = \left\{
\mathbb{P} \in \mathcal{P}_0\left(\mathbb{R}^I\times\mathbb{R}\times [S]\right) \left|
\begin{array}
~(\tilde{\pmb{z}}, \tilde{u}, \tilde{s}) \in \mathbb{P} &\\
\mathbb{E}_{\mathbb{P}}\left[\tilde{u} | \tilde{s} \in [S]\right] \leq \theta & \\
\mathbb{P}\left[\left.(\pmb{z}, u)\in\mathcal{Z}_s ~\right| \tilde{s} = s\right] = 1, & \forall s \in [S] \\
\mathbb{P}\left[\tilde{s} = s\right] = \frac{1}{S} &
\end{array}
\right.
\right\}
\end{align}
$$

with \\(\theta \geq 0\\) the parameter capturing the distance between the distribution \\(\mathbb{P}\\) and the empirical distribution \\(\hat{\pmb{z}}\\). The support \\(\mathcal{Z}_s\\) for each sample \\(s\\) is defined as

$$
\mathcal{Z}_s = \left\{ (\pmb{z}, u): \pmb{0} \leq \pmb{z} \leq \overline{\pmb{z}}, \|\pmb{z} - \hat{\pmb{z}}_s \|_2 \leq u
\right\}.
$$

In this numerical experiment, parameters of the model and the ambiguity set are specified as follows:

- The number of products: \\(I=2\\);
- Sample size: \\(S=50\\);
- The unit cost of each product: \\(c_i=1\\)
- The unit price of each product: \\(p_i\\) is randomly generatd from a uniform distribution on \\([1, 5]\\).
- The upper bound of demand: the array \\(\overline{\pmb{z}}\\) is randomly generated from the uniform distribution on \\([0, 100]^I\\);
- The sample data of demands: the array \\(\hat{\pmb{z}}\in\mathbb{R}^{S\times I}\\) is randomly generated from the uniform distribution on \\([\pmb{0}, \overline{\pmb{u}}]\\);
- The budget \\(d=50 I\\);
- The Wasserstein metric parameter \\(\theta=0.01 \min\\{\overline{\pmb{z}}\\}\\).

The RSOME code for implementing the model above is given as follows.

```python
from rsome import dro
from rsome import norm
from rsome import E
from rsome import grb_solver as grb
import numpy as np
import numpy.random as rd

# model and ambiguity set parameters
I = 2
S = 50
c = np.ones(I)
d = 50 * I
p = 1 + 4*rd.rand(I)
zbar = 100 * rd.rand(I)
zhat = zbar * rd.rand(S, I)
theta = 0.01 * zbar.min()

# modeling with RSOME
model = dro.Model(S)                        # create a DRO model with S scenarios
z = model.rvar(I)                           # random demand z
u = model.rvar()                            # auxiliary random variable

fset = model.ambiguity()                    # create an ambiguity set
for s in range(S):
    fset[s].suppset(0 <= z, z <= zbar,
                    norm(z - zhat[s]) <= u) # define the support for each scenario
fset.exptset(E(u) <= theta)                 # the Wasserstein metric constraint
pr = model.p                                # an array of scenario probabilities
fset.probset(pr == 1/S)                     # support of scenario probabilities

x = model.dvar(I)                           # define first-stage decisions
y = model.dvar(I)                           # define decision rule variables
y.adapt(z)                                  # y affinely adapts to z
y.adapt(u)                                  # y affinely adapts to u
for s in range(S):
    y.adapt(s)                              # y adapts to each scenario s

model.minsup(-p@x + E(p@y), fset)           # worst-case expectation over fset
model.st(y >= 0)                            # constraints
model.st(y >= x - z)                        # constraints
model.st(x >= 0)                            # constraints
model.st(c@x == d)                          # constraints

model.solve(grb)                            # solve the model by Gurobi
```

```
Being solved by Gurobi...
Solution status: 2
Running time: 0.0271s
```

<br>

#### Reference

<a id="ref1"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.

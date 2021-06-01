<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Distributionally robust optimization for medical appointment scheduling

In this example,  we consider a medical appointment scheduling problem described in [Adaptive distributionally robust optimization](http://www.optimization-online.org/DB_FILE/2016/03/5353.pdf), where \\(N\\) patients arrive at their stipulated schedule and may have to wait in a queue to be served by a physician. The patients’ consultation times are uncertain and their arrival schedules are determined at the first stage, which can influence the waiting times of the patients and the overtime of the physician.

A distributionally robust optimization model presented below is used to minimize the worst-case expected total cost of patients waiting and physician overtime over a partial cross moment ambiguity set.

$$
\begin{align}
\min~&\sup\limits_{\mathbb{P}\in\mathcal{F}}~\mathbb{E}_{\mathbb{P}}\left[\sum\limits_{j=1}^N y_i(\tilde{\pmb{z}}, \tilde{\pmb{u}}) + \gamma y_{N+1}(\tilde{\pmb{z}}, \tilde{\pmb{u}})\right] && \\
\text{s.t.}~&y_j(\pmb{z}, \pmb{u}) - y_{j-1}(\pmb{z}, \pmb{u}) + x_{j+1} \geq z_{j-1} && \forall (\pmb{z}, \pmb{u})\in \mathcal{Z}, ~\forall j \in \{2, 3, ..., N+1\} \\
&\pmb{y}(\pmb{z}, \pmb{u}) \geq \pmb{0}, && \forall (\pmb{z}, \pmb{u})\in \mathcal{Z} \\
&\pmb{x} \geq \pmb{0},~\sum\limits_{j=1}^N x_j \leq T && \\
\end{align}
$$

with the random variable \\(\tilde{z}\_j\\) representing the uncertain consultation time of each patient. The first-stage decision \\(x_j\\) denotes the inter-interval time between patient \\(j\\) to the adjacent patient \\(j+1\\), for \\(j\in[N-1]\\), and \\(x_N\\) is the arrival of the last patient and the scheduled completion time for the physician before overtime commences. The second-stage decision \\(y_j\\) denotes the waiting time of patient \\(j\\), with \\(i \in [N]\\) and \\(y_{N+1}\\) represents the overtime of the physician. In order to achieve a tractable formulation, it is approximated by the decision rule \\(\pmb{y}(\pmb{z}, \pmb{u})\\) which affinely adapts to random variables \\(\pmb{z}\\) and auxiliary variables \\(\pmb{u}\\).

In this numerical experiment, we consider the lifted form of the partial cross moment ambiguity set \\(\mathcal{F}\\) given below.

$$
\begin{align}
\mathcal{F} =
\left\{
\mathbb{P}\in\mathcal{P}(\mathbb{R}^N\times\mathbb{R}^{N+1})
\left|
\begin{array}
~(\tilde{\pmb{z}}, \tilde{\pmb{u}}) \sim \mathbb{P} \\
\pmb{z} \geq \pmb{0} & \\
(z_j - \mu_j)^2 \leq u_j, & \forall j \in [N] \\
(\pmb{1}^T(\pmb{z} - \pmb{\mu}))^2 \leq u_{N+1} \\  
\mathbb{E}_{\mathbb{P}}(\tilde{\pmb{z}}) = \pmb{\mu} & \\
\mathbb{E}_{\mathbb{P}}(\tilde{u}_j) \leq \sigma_j, & \forall j \in [N] \\
\mathbb{E}_{\mathbb{P}}(\tilde{u}_N) \leq \pmb{1}^T\pmb{\Sigma}\pmb{1}
\end{array}
\right.
\right\}.
\end{align}
$$

Values of model and ambiguity set parameters are specified as follows:
- Number of patients: \\(N=8\\),
- Unit cost of physician overtime: \\(\gamma=2\\),
- Correlation parameter: \\(\alpha=0.25\\),
- Expected consultation times: \\(\pmb{\mu}\\) are random numbers uniformly distributed between \\([30, 60]\\),
- Standard deviations of the consultation times: \\(\sigma_j=\mu_j\epsilon_j\\), where \\(\epsilon_j\\) is a random number uniformly distributed between [0, 0.3],
- The scheduled completion time for the physician before overtime commences:

$$
T=\sum\limits_{j=1}^N\mu_j + 0.5\sqrt{\sum\limits_{j=1}^N\sigma_j^2},
$$

- Covariance matrix: \\(\pmb{\Sigma}\\), with its elements to be:

$$
\left[\pmb{\Sigma}\right]_{ij} =
\begin{cases}
\alpha \sigma_i\sigma_j & \text{if }i\not=j \\
\sigma_j^2 & \text{otherwise}.
\end{cases}
$$

The RSOME code for implementing this distributionally robust optimization model is given below.

```python
from rsome import dro
from rsome import square
from rsome import E
from rsome import grb_solver as grb
import numpy as np
import numpy.random as rd


# Model and ambiguity set parameters
N = 8
gamma = 2
alpha = 0.25
mu = 30 + 30*rd.rand(N)
sigma = mu * 0.3*rd.rand(1, N)
T = mu.sum() + 0.5*((sigma**2).sum())**0.5

mul = np.diag(np.ones(N))*(1-alpha) + np.ones((N, N))*alpha
Sigma = (sigma.T @ sigma) * mul

# Modeling with RSOME
model = dro.Model()                                  # Define a DRO model
z = model.rvar(N)                                    # Random variable z
u = model.rvar(N+1)                                  # Auxiliary variable u

fset = model.wks(z >= 0, square(z - mu) <= u[:-1],
                 square((z-mu).sum()) <= u[-1],      # Support of random variables
                 E(z) == mu, E(u[:-1]) <= sigma**2,
                 E(u[-1]) <= Sigma.sum())            # Support of expectations

x = model.dvar(N)                                    # The first-stage decision
y = model.dvar(N+1)                                  # The decision rule
y.adapt(z)                                           # Define affine adaptation
y.adapt(u)                                           # Define affine adaptation

model.minsup(E(y[:-1].sum() + gamma*y[-1]), fset)    # Worst-case expected cost
model.st(y[1:] - y[:-1] + x >= z)                    # Constraints
model.st(y >= 0)                                     # Constraints
model.st(x >= 0)                                     # Constraints
model.st(x.sum() <= T)                               # Constraints

model.solve(grb)                                     # Solve the model by Gurobi
```

```
Being solved by Gurobi...
Solution status: 2
Running time: 0.0516s
```

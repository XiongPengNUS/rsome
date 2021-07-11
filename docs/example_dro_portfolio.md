<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Distributionally Robust Portfolio

Given \\(J\\) investment options whose random returns are collectively denoted by a random vector \\(\tilde{\pmb{z}}\\), we choose a portfolio that maximizes the worst-case (<i>i.e.</i>, least) expected utility:

$$
\begin{align}
\max_{\pmb{x}}~& \inf_{\mathbb{P} \in \mathcal{F}}\mathbb{E}_\mathbb{P}\left[U\left(\tilde{\pmb{d}}^{\top}\pmb{x}\right)\right] \\
\text{s.t.} ~& \pmb{e}^{\top}\pmb{x} = 1 \\
& \pmb{x} \in \mathbb{R}^J_+.
\end{align}
$$

Here we assume that a commonly used utility function is piecewise affine and concave such that \\(U(y) = \min_{k \in [K]} \left\\{\alpha_k y + \beta_k\right\\}\\) with \\(\alpha_k \geq 0\\). We also assume the mean returns, the variance of each option, as well as the variance of sum of these options are known, captured by the widely used partial cross-moment ambiguity set ([Bertsimas et al. (2019)](#ref1), [Delgage and Ye (2010)](#ref3),  [Wiesemann et al. (2014)](#ref4)):

$$
\mathcal{G} =
	\left\{
	\mathbb{P} \in \mathcal{P}_0 (\mathbb{R}^J)
	~\left|~
	\begin{array}{l@{\quad}l}
		\tilde{\pmb{d}} \sim \mathbb{R} \\
		\mathbb{E}_\mathbb{P}[\tilde{\pmb{d}}] = \pmb{\mu} \\
		\mathbb{E}_\mathbb{P}[(\tilde{d}_j - \mu_j)^2] \leq \sigma^2_j & \forall j \in [J] \\
		\mathbb{E}_\mathbb{P}[(\pmb{e}^\top(\tilde{\pmb{d}}-\pmb{\mu}))^2] \leq \pmb{e}^\top\pmb{\Sigma}\pmb{e} \\
		\mathbb{P}[\tilde{\pmb{d}} \in [\underline{\pmb{d}}, \bar{\pmb{d}}]] = 1
	\end{array}
	\right.
	\right\},
$$

where \\(\pmb{\mu} = (\mu_j)\_{j \in [J]}\\) is the vector of mean values, \\(\pmb{\sigma} = (\sigma_j)_{j \in [J]}\\) is the vector of standard deviations, and \\(\pmb{\Sigma}\\) is the covariance matrix of random returns. Introducing an auxiliary random vector \\(\tilde{\pmb{u}}\\) and auxiliary random scenario \\(\tilde{s}\\), the above ambiguity set can be reformulated into the format of event-wise ambiguity set with only one scenario (<i>i.e.</i>, \\(S = 1\\)):
$$
\mathcal{F} =
\left\{
	\mathbb{P} \in \mathcal{P}_0 (\mathbb{R}^J \times \mathbb{R}^{J+1} \times \{1\}) ~\left|~
	\begin{array}{ll}
		(\tilde{\pmb{d}}, \tilde{\pmb{u}}, \tilde{s}) \sim \mathbb{P} \\
		\mathbb{E}_\mathbb{P}[\tilde{\pmb{d}} \mid \tilde{s} = 1] = \pmb{\mu} \\
		\mathbb{E}_\mathbb{P}[\tilde{u}_j \mid \tilde{s} = 1] \leq \sigma^2_j &~\forall j \in [J] \\
		\mathbb{E}\_\mathbb{P}[\tilde{u}_{J+1} \mid \tilde{s} = 1] \leq \pmb{e}^\top\pmb{\Sigma}\pmb{e} \\
		\mathbb{P}[(\tilde{\pmb{d}}, \tilde{\pmb{u}}) \in \mathcal{Z}_1 \mid \tilde{s} = 1] = 1 \\
		\mathbb{P}[\tilde{s} = 1] = 1
	\end{array}
	\right.
	\right\},
$$

where the support set is given by

$$
	\mathcal{Z}_1 = \left\{(\pmb{d}, \pmb{u}) \in \mathbb{R}^J \times \mathbb{R}^{J+1}: \; \pmb{d} \in [\underline{\pmb{d}}, \bar{\pmb{d}}], \; (\pmb{e}^\top(\pmb{d}-\pmb{\mu}))^2 \leq u_{J+1}, \; (d_j -\mu_j)^2 \leq u_j ~\forall j \in [J]\right\}.
$$

These two ambiguity sets are equivalent in the sense that \\(\mathcal{G} = \cup_{\mathbb{P} \in \mathcal{F}}\{\Pi_{\tilde{\pmb{d}}}\mathbb{P}\}\\), where \\(\Pi_{\tilde{\pmb{d}}}\mathbb{P}\\) denotes the marginal distribution of \\(\tilde{\pmb{d}}\\) under a joint distribution \\(\mathbb{P}\\) of \\((\tilde{\pmb{d}}, \tilde{\pmb{u}}, \tilde{s})\\). With this equivalence and the fact that its objective function is independent of the auxiliary uncertainty \\((\tilde{\pmb{u}}, \tilde{s})\\), the distributionally robust optimization problem above is equivalent to

$$
\begin{align}
\max_{\pmb{x}} ~& \inf_{\mathbb{P} \in \mathcal{F}}\mathbb{E}_\mathbb{P}\left[U\left(\tilde{\pmb{d}}^{\top}\pmb{x}\right)\right] \\
\text{s.t.} ~& \pmb{e}^\top\pmb{x} = 1 \\
& \pmb{x} \in \mathbb{R}^J_+.
\end{align}
$$

By applying <b>Theorem 1</b> in [Chen et al. (2020)](#ref2), we obtain the following equivalent adaptive problem in our distributionally robust optimization framework:

$$
\begin{align}
\max_{\pmb{x}} ~ & \inf_{\mathbb{P} \in \mathcal{F}}\mathbb{E}_\mathbb{P}\left[y\left(1, \tilde{\pmb{d}}, \tilde{\pmb{u}}\right)\right] \\
\text{s.t.} ~  & y(1, \pmb{d}, \pmb{u}) \geq \alpha_k\cdot (\pmb{d}^\top\pmb{x}) + \beta_k & \forall (\pmb{d}, \pmb{u}) \in \mathcal{Z}_1, \; k \in [K] \\
& y \in \overline{\mathcal{A}}(\{\{1\}\}, [J]) \\
& \pmb{e}^\top\pmb{x} = 1 \\
& \pmb{x} \in \mathbb{R}^J_+,
\end{align}
$$

where we would like to highlight the introduction of an auxiliary one-dimensional wait-and-see decision \\(y\\), which obeys the event-wise affine adaptation \\(\overline{\mathcal{A}}(\{\{1\}\}, [2J+1])\\) that affinely depends on all random components \\(\tilde{d}\_1,\dots,\tilde{d}\_J\\) as well as \\(\tilde{u}_1,\dots,\tilde{u}\_{J+1}\\).

As a concrete example, we consider $J=6$ stocks (TSLA, AAPL, AMZN, GOOG, ZM, and FB) in the year of 2020 and collect daily returns using the code segment below.

```python
import pandas as pd
import pandas_datareader.data as web

stocks = ['TSLA', 'AAPL', 'AMZN', 'GOOG', 'ZM', 'FB']
start = '1/1/2020'              # starting date of historical data
end='1/1/2021'                  # end date of historical data

data = pd.DataFrame([])
for stock in stocks:
    each = web.DataReader(stock, 'yahoo', start=start, end=end)
    close = each['Adj Close'].values
    returns = (close[1:] - close[:-1]) / close[:-1]
    data[stock] = returns
```

Paramters of the utility function are given below
- Number of pieces of the utility function \\(U(y)\\): \\(K=3\\).
- Linear terms of the piecewise utility function \\(U(y)\\): \\(\alpha=(6, 3, 1)\\).
- Constant terms of the piecewise utility function \\(U(y)\\): \\(\beta=(0.02, 0, 0)\\).
<br>

Parameters of the utility function and the ambiguity set are defined by the code segment below.

```python
import numpy as np

alpha = [6, 3, 1]               # linear terms of the piecewise utility function
beta = [0.02, 0, 0]             # constant terms of the piecewise utility function

mu = data.mean().values         # the expected values of stock returns
sigma = data.sigma().values     # variance of stock returns
Sigma = data.cov().values       # covariance matrix of stock returns
z_ub = data.max().values        # upper bounds of stock returns
z_lb = data.min().values        # lower bounds of stock returns

J = len(mu)                     # number of stocks
K = len(alpha)                  # number of pieces of the utility function
```
The portfolio optimization problem is then formulated and solved by the code segment below.

```python
from rsome import dro                        # import the dro module
from rsome import E                          # import the notion of expectation
from rsome import square                     # import the square function
from rsome import grb_solver as grb          # import the interface for Gurobi

model = dro.Model()                          # create an RSOME DRO model

d = model.rvar(J)                            # J random variables as an array d
u = model.rvar(J+1)                          # J+1 random variables as an array u
fset = model.ambiguity()                     # create an ambiguity set
fset.suppset(d >= d_lb, d <= d_ub,
             square(d-mu) <= u[:-1],
             square((d-mu).sum()) <= u[-1])  # support of random variables
fset.exptset(E(d) == mu,
             E(u[:-1]) <= sigma**2,
             E(u[-1]) <= Sigma.sum())        # uncertainty set of expectations

x = model.dvar(J)                            # J decision variables as an array x
y = model.dvar()                             # 1 decision variable as y
y.adapt(d)                                   # y affinely adapts to d
y.adapt(u)                                   # y affinely adapts to u

model.maxinf(E(y), fset)                     # worst-case expectation over fset
model.st(y <= a*(d@x) + b
         for a, b in zip(alpha, beta))       # piecewise linear constraints
model.st(x.sum() == 1, x >= 0)               # constraints of x

model.solve(grb)                             # solve the model by Gurobi
```

Different from the `ro` module, the `dro` module enables users to define an ambiguity set for capturing the ambiguous distribution. The ambiguity set is created by calling the `ambiguity()` method and then the support of random variables `d` and `u` and the uncertainty set of their expectations are specified by the `suppset()` and `exptset()`, respectively. Similar  to the `ro` module, the affine adaptation of decision `y` upon random variables is defined using the `adapt()` method. The objective as the worst-case expectation over the ambiguity set `fset` is then defined by the `maxinf()` method, which means maximizing the infimum of the expected term.


#### Reference

<a id="ref1"></a>

Bertsimas, Dimitris, Melvyn Sim, and Meilin Zhang. 2019. [Adaptive distributionally robust optimization](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2017.2952). <i>Management Science</i> <b>65</b>(2) 604-618.

<a id="ref2"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.

<a id="ref3"></a>

Delage, Erick, and Yinyu Ye. 2010. [Distributionally robust optimization under moment uncertainty with application to data-driven problems](https://pubsonline.informs.org/doi/abs/10.1287/opre.1090.0741). <i>Operations Research</i> <b>58</b>(3) 595-612.

<a id="ref4"></a>

Wiesemann, Wolfram, Daniel Kuhn, and Melvyn Sim. 2014. [Distributionally robust convex optimization](https://pubsonline.informs.org/doi/abs/10.1287/opre.2014.1314). <i>Operations Research</i> <b>62</b>(6) 1358-1376.

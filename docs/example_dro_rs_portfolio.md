<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Robust Satisficing for Portfolio Optimization

In this example, we consider the portfolio optimization problem introduced in [Long et al. (2022)](#ref1), where the decision-maker invests in \\(N\\) risky assets and the portfolio risk is evaluated using the empirical distribution \\(\hat{\mathbb{P}}\\) constructed from historical data \\((\hat{z}_1,...,\hat{z}_S)\\). Here, the problem is solved as a robust satisficing model

$$
\begin{align}
\min ~& k_0 + wk_1 \\
\text{s.t.}~& \mathbb{E}_{\mathbb{P}}\left[\pmb{x}^{\top}\tilde{\pmb{z}}\right] \geq \tau - k_0 \Delta_W(\mathbb{P}, \hat{\mathbb{P}}) &\forall \mathbb{P} \in \mathcal{P}_0 (\mathcal{Z}) \\
&\alpha + \frac{1}{\epsilon}\mathbb{E}_{\mathbb{P}} \left[(-\pmb{x}^{\top}\tilde{\pmb{z}} - \alpha)^+\right] \leq \beta + k_1 \Delta_W(\mathbb{P}, \hat{\mathbb{P}}) &\forall \mathbb{P} \in \mathcal{P}_0 (\mathcal{Z}) \\
& \pmb{1}^{\top} \pmb{x} = 1 \\
& \pmb{x} \in \mathbb{R}_+^N, \alpha \in \mathbb{R},
\end{align}
$$

for some target \\(\tau\\) of the expected return, and some target \\(\beta\\) of the \\(1-\epsilon\\) conditional value at risk (CVaR). Here, the distribution of the random return \\(\tilde{\pmb{z}}\\) is denoted by \\(\mathbb{P}\\), and the Wasserstein distance with one-norm between \\(\mathbb{P}\\) and the empirical distribution \\(\hat{\mathbb{P}}\\) is denoted by \\(\Delta_W(\mathbb{P}, \hat{\mathbb{P}})\\). Detailed information on model parameters are provided below

- The sample size of the empirical dataset \\(S=300\\);
- The number of assets \\(n=10\\);
- The penalty parameter \\(w=10\\);
- The confidence level \\(1-\epsilon=95\%\\);
- The target of the expected return \\(\tau=0.125\\);
- The target of the CVaR \\(\beta = 0.10\\);
- The empirical dataset \\(\hat{\pmb{z}}\\) used to represent the distribution of the random return \\(z_i = \varphi + \zeta_i\\), where \\(\varphi \sim \mathcal{N}(0, 5\%)\\) and \\(\zeta_i \sim \mathcal{N}(i\times 2\%, i\times 3\%)\\);

and these parameters are defined by the following code segment. 

```python
import numpy as np

s, n = 300, 10
w = 10
beta = 0.10
epsilon = 0.05
tau = 0.125

np.random.seed(1)

i = np.arange(1, n+1)
phi = 0.05 * np.random.normal(size=(s, n))
zeta = 0.02*i + 0.03*i*np.random.normal(size=(s, n))
zhat = phi + zeta
```

#### Deterministic Equivalent Problem

According to <b>Theorem 9</b> of [Long et al. (2022)](#ref1), the robust satsificing model can be reformulated as the deterministic optimization problem below:

$$
\begin{align}
\min ~&\|\pmb{x}\|_{\infty} \\
\text{s.t.}~&\frac{1}{S}\sum\limits_{s\in[S]}y_{1s} \geq \tau \\
&y_{1s} \leq \pmb{x}^{\top}\hat{\pmb{z}}_s & \forall s\in [S] \\
&\alpha + \frac{1}{\epsilon S}\sum\limits_{s\in[S]} y_{2s} \leq \beta \\
&y_{2s} \geq -\pmb{x}^{\top}\hat{\pmb{z}}_s - \alpha & \forall s\in [S] \\
&y_{2s} \geq 0 &\forall s \in [S] \\
&\pmb{1}^{\top}\pmb{x} = 1 \\
&\pmb{x}\in\mathbb{R}_+^{N}, \alpha \in \mathbb{R}.
\end{align}
$$

Such a deterministic optimization can be implemented using the following code. 

```python
from rsome.rsome import ro
import rsome.rsome as rso

model = ro.Model()

x = model.dvar(n)
alpha = model.dvar()
y1 = model.dvar(s)
y2 = model.dvar(s)

model.min(rso.norm(x, 'inf'))
model.st((1/s) * y1.sum() >= tau)
model.st(alpha + (1/s/epsilon)*y2.sum() <= beta)
model.st(y2 >= 0)
model.st(x >= 0, x.sum() == 1)
for j in range(s):
    model.st(y1[j] <= x@zhat[j])
    model.st(y2[j] >= -x@zhat[j] - alpha)

model.solve()
```

```
Being solved by the default LP solver...
Solution status: 0
Running time: 0.0203s
```

The optimal solution of `x` as the allocation of capital is shown below.

```python
print(x.get())
```

```
[0.         0.08771868 0.11403517 0.11403517 0.11403517 0.11403517
 0.11403517 0.11403517 0.11403517 0.11403517]
```

#### Scenario-wise Robust Optimization Problem

The robust satisficing model above is also equivalent to the following scenario-wise robust optimization problem:

$$
\begin{align}
\min ~& k_0 + wk_1 \\
\text{s.t.}~& \frac{1}{S}\sum\limits_{s\in[S]}v_{1s} \geq \tau \\
&\alpha + \frac{1}{S}\sum\limits_{s\in[S]}v_{2s} \leq \beta  \\
&v_{1s} \leq \pmb{z}^{\top}\pmb{x} + k_0u & \forall (\pmb{z}, u)\in\mathcal{Z}_s \\
&v_{2s} \geq \frac{1}{\epsilon}(-\pmb{z}^{\top}\pmb{x}-\alpha) - k_1u & \forall (\pmb{z}, u)\in\mathcal{Z}_s \\
&v_{2s} \geq - k_1u & \forall (\pmb{z}, u)\in\mathcal{Z}_s \\
&\pmb{1}^{\top} \pmb{x} = 1 \\
&\pmb{x} \in \mathbb{R}_+^N, \alpha \in \mathbb{R}, \pmb{v} \in \mathbb{R}^{2\times S},
\end{align}
$$

with the scenario-wise support \\(\mathcal{Z}_s = \left\\{(\pmb{z}, u): ~\|\pmb{z} - \hat{\pmb{z}}_s\|_1 \leq u \right\\}\\). The robust model can be implemented by the code segment below.

```python
from rsome.rsome import ro
import rsome.rsome as rso

model = ro.Model()

x = model.dvar(n)
alpha = model.dvar()
k0 = model.dvar()
k1 = model.dvar()
v1 = model.dvar(s)
v2 = model.dvar(s)

z = model.rvar(n)
u = model.rvar()

r = z @ x
model.min(k0 + w*k1)
model.st(v1.sum() * (1/s) >= tau)
model.st(alpha + v2.sum() * (1/s) <= beta)
for j in range(s):
    uset = (rso.norm(z - zhat[j], 1) <= u)
    model.st((v1[j] <= r + k0*u).forall(uset))
    model.st((v2[j] >= (1/epsilon)*(-r-alpha) - k1*u).forall(uset))
    model.st((v2[j] >= -k1*u).forall(uset))
model.st(x >= 0, x.sum() == 1)
model.solve()
```

```
Being solved by the default LP solver...
Solution status: 0
Running time: 0.6255s
```

The optimal solution of the robust model is the same as the deterministic problem. 

```python
print(x.get())
```

```
[-0.          0.08771868  0.11403517  0.11403517  0.11403517  0.11403517
  0.11403517  0.11403517  0.11403517  0.11403517]
```

```python
model.get()
```

```
22.921068194932346
```

#### Distributionally Robust Model

The robust satisficing model can be implemented directly using the `rsome.dro` modeling environment, see the code below.

```python
from rsome.rsome import dro
from rsome.rsome import E
import rsome.rsome as rso

model = dro.Model(s)
    
x = model.dvar(n)
alpha = model.dvar()
k0 = model.dvar()
k1 = model.dvar()
    
z = model.rvar(n)
u = model.rvar()
fset = model.ambiguity()
for j in range(s):
    fset[j].suppset(rso.norm(z-zhat[j], 1) <= u)
pr = model.p
fset.probset(pr == 1/s)
    
r = z @ x
model.minsup(k0 + w*k1, fset)
model.st(E(r) >= tau - E(k0*u))
model.st(alpha + (1/epsilon)*E(rso.maxof(-r-alpha, 0)) <= beta + E(k1*u))
model.st(x >= 0, x.sum() == 1)
    
model.solve()
```

Here, we use the scenario-wise ambiguity set proposed in [Chen et al. (2020)](#ref2) to capture the Wasserstein ball, the same optimal solution can be obtained. 

```python
print(x.get())
```

```
[-0.          0.08771868  0.11403517  0.11403517  0.11403517  0.11403517
  0.11403517  0.11403517  0.11403517  0.11403517]
```

```python
model.get()
```

```
22.92106819493208
```

#### Reference

<a id="ref1"></a>

Long, Daniel Zhuoyu, Melvyn Sim, and Minglong Zhou. 2022. [Robust Satisficing](https://pubsonline.informs.org/doi/abs/10.1287/opre.2021.2238). <i>Operations Research</i> <b>71</b>(1):61-82.

<a id="ref2"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.
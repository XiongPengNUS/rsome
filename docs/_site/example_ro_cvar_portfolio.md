<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Conditional value-at-risk with application to robust portfolio management

This robust portfolio management model is proposed by [Zhu and Fukushima (2009)](#ref1). The portfolio allocation is determined via minimizing the worst-case conditional value-at-risk (CVaR) under ambiguous distribution information. The generic formulation is given as

$$
\begin{align}
\min~&\max\limits_{\pmb{\pi}\in \Pi} \alpha + \frac{1}{1-\beta}\pmb{\pi}^T\pmb{u} &\\
\text{s.t.}~& u_k \geq \pmb{y}_k^T\pmb{x} - \alpha, &\forall k = 1, 2, ..., s \\
&u_k \geq 0, &\forall k=1, 2, ..., s \\
&\sum\limits_{k=1}^s\pi_k\pmb{y}_k^T\pmb{x} \geq \mu, &\forall \pmb{\pi} \in \Pi  \\
&\underline{\pmb{x}} \leq \pmb{x} \leq \overline{\pmb{x}} \\
&\pmb{1}^T\pmb{x} = w_0
\end{align}
$$

with investment decisions\\(\pmb{x}\in\mathbb{R}^n\\) and auxiliary variables \\(\pmb{u}\in\mathbb{R}^s\\) and \\(\alpha\in\mathbb{R}\\), where \\(n\\) is the number of stocks, and \\(s\\) is the number of samples. The array \\(\pmb{\pi}\\) denotes the probabilities of samples, and \\(\Pi\\) is the uncertainty set that captures the distributional ambiguity of probabilities. The constant array \\(\pmb{y}_k\in\mathbb{R}^n\\) indicates the \\(k\\)th sample of stock return rates, and \\(\bar{x}\\) and \\(\underline{x}\\) are the lower and upper bounds of \\(x\\). The worst-case minimum expected overall return rate is set to be \\(\mu=0.001\\), the confidence level is \\(\beta=0.95\\), and the budget of investment is set to be \\(w_0=1\\). In this case study, we consider the sample data of five stocks "JPM", "AMZN", "TSLA", "AAPL", and	"GOOG" in the year of 2020, and the other parameters are specified by the following code segment.

```python
import pandas as pd
import pandas_datareader.data as web
import numpy as np

stocks = ['JPM', 'AMZN', 'TSLA', 'AAPL', 'GOOG']
start = '1/1/2020'              # starting date of historical data
end='12/31/2020'                # end date of historical data

data = pd.DataFrame([])
for stock in stocks:
    each = web.DataReader(stock, 'yahoo', start=start, end=end)
    close = each['Adj Close'].values
    returns = (close[1:] - close[:-1]) / close[:-1]
    data[stock] = returns

data
```

<div>
<table border="1" class="dataframe mystyle">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JPM</th>
      <th>AMZN</th>
      <th>TSLA</th>
      <th>AAPL</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.012124</td>
      <td>0.027151</td>
      <td>0.028518</td>
      <td>0.022816</td>
      <td>0.022700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.013197</td>
      <td>-0.012139</td>
      <td>0.029633</td>
      <td>-0.009722</td>
      <td>-0.004907</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.000795</td>
      <td>0.014886</td>
      <td>0.019255</td>
      <td>0.007968</td>
      <td>0.024657</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.017001</td>
      <td>0.002092</td>
      <td>0.038801</td>
      <td>-0.004703</td>
      <td>-0.000624</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.007801</td>
      <td>-0.007809</td>
      <td>0.049205</td>
      <td>0.016086</td>
      <td>0.007880</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>248</th>
      <td>-0.004398</td>
      <td>-0.003949</td>
      <td>0.024444</td>
      <td>0.007712</td>
      <td>0.003735</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.006585</td>
      <td>0.035071</td>
      <td>0.002901</td>
      <td>0.035766</td>
      <td>0.021416</td>
    </tr>
    <tr>
      <th>250</th>
      <td>-0.002633</td>
      <td>0.011584</td>
      <td>0.003465</td>
      <td>-0.013315</td>
      <td>-0.009780</td>
    </tr>
    <tr>
      <th>251</th>
      <td>0.002800</td>
      <td>-0.010882</td>
      <td>0.043229</td>
      <td>-0.008527</td>
      <td>-0.010917</td>
    </tr>
    <tr>
      <th>252</th>
      <td>0.013641</td>
      <td>-0.008801</td>
      <td>0.015674</td>
      <td>-0.007703</td>
      <td>0.007105</td>
    </tr>
  </tbody>
</table>
<p>253 rows × 5 columns</p>
</div>

```python
y = data.values     # stock data as an array
s, n = y.shape      # s: sample size; n: number of stocks

x_lb = np.zeros(n)  # lower bounds of investment decisions
x_ub = np.ones(n)   # upper bounds of investment decisions

beta =0.95          # confidence interval
w0 = 1              # investment budget
mu = 0.001          # target minimum expected return rate
```

#### Nominal CVaR model

In the nominal model, the CVaR and expected returns are evaluated assuming the exact distribution of stock returns is accurately represented by the historical samples without any distributional ambiguity. In other words, \\(\Pi\\) is written as a singleton uncertainty \\(\Pi = \\{\pmb{\pi}^0 \\}\\), where \\(\pi_k^0=1/s\\), with \\(k=1, 2, ..., s\\). The Python code for implementing the nominal model is given below.

```python
from rsome import ro
from rsome import grb_solver as grb

model = ro.Model()

pi = np.ones(s) / s         # no ambiguity in probability distribution

x = model.dvar(n)
u = model.dvar(s)
alpha = model.dvar()

model.min(alpha + 1/(1-beta) * (pi@u))
model.st(u >= y@x - alpha)
model.st(u >= 0)
model.st(pi@y@x >= mu)
model.st(x >= x_lb, x <= x_ub, x.sum() == w0)

model.solve(grb)
```

```
Being solved by Gurobi...
Solution status: 2
Running time: 0.0073s
```

The portfolio decision for the nominal model is retrieved by the following code.

```python
x.get().round(4)    # the optimal portfolio decision with 4 d.p.
```

```
array([0.3135, 0.5331, 0.0455, 0.    , 0.1079])
```

#### Worst-case CVaR model with box uncertainty
Now we consider a box uncertainty set

$$
\Pi = \left\{\pmb{\pi}: \pmb{\pi} = \pmb{\pi}^0 + \pmb{\eta}, \pmb{1}^T\pmb{\eta}=0, \underline{\pmb{\eta}}\leq \pmb{\eta} \leq \bar{\pmb{\eta}} \right\}.
$$

In this case study, we assume that \\(-\underline{\pmb{\eta}}=\bar{\pmb{\eta}}=0.0001\\), and the Python code for implementation is provided below.

```python
model = ro.Model()

eta_ub = 0.0001                 # upper bound of eta
eta_lb = -0.0001                # lower bound of eta

eta = model.rvar(s)             # eta as random variables
uset = (eta.sum() == 0,
        eta >= eta_lb,
        eta <= eta_ub)
pi = 1/s + eta                  # pi as inexact probabilities

x = model.dvar(n)
u = model.dvar(s)
alpha = model.dvar()

model.minmax(alpha + 1/(1-beta) * (pi@u), uset)
model.st(u >= y@x - alpha)
model.st(u >= 0)
model.st(pi@y@x >= mu)
model.st(x >= x_lb, x <= x_ub, x.sum() == w0)

model.solve(grb)
```

```
Being solved by Gurobi...
Solution status: 2
Running time: 0.0268s
```

```python
x.get().round(4)    # the optimal portfolio decision with 4 d.p.
```

```
array([0.2605, 0.5601, 0.0651, 0.    , 0.1143])
```

#### Worst-case CVaR model with ellipsoidal uncertainty

In cases that \\(\Pi\\) is an ellipsoidal uncertainty set

$$
\Pi = \left\{\pmb{\pi}: \pmb{\pi} = \pmb{\pi}^0 + \rho\pmb{\eta}, \pmb{1}^T\pmb{\eta}=0, \pmb{\pi}^0 + \rho\pmb{\eta} \geq \pmb{0}, \|\pmb{\eta}\| \leq 1 \right\},
$$

where the nominal probability \\(\pmb{\pi}^0\\) is the center of the ellipsoid, and the constant \\(\rho=0.001\\), then the model can be implemented by the code below.

```python
model = ro.Model()

rho = 0.001

eta = model.rvar(s)
uset = (eta.sum() == 0, 1/s + rho*eta >= 0,
        rso.norm(eta) <= 1)
pi = 1/s + rho*eta

x = model.dvar(n)
u = model.dvar(s)
alpha = model.dvar()

model.minmax(alpha + 1/(1-beta) * (pi@u), uset)
model.st(u >= y@x - alpha)
model.st(u >= 0)
model.st(pi@y@x >= mu)
model.st(x >= x_lb, x <= x_ub, x.sum() == w0)

model.solve(grb)
```

```
Being solved by Gurobi...
Solution status: 2
Running time: 0.0285s
```

```python
x.get().round(4)    # the optimal portfolio decision with 4 d.p.
```

```
array([0.1702, 0.6098, 0.0255, 0.    , 0.1945])
```

<br>
#### Reference

<a id="ref1"></a>

Zhu, Shushang, and Masao Fukushima. "[Worst-case conditional value-at-risk with application to robust portfolio management](https://pubsonline.informs.org/doi/abs/10.1287/opre.1080.0684)." <i>Operations research</i> 57.5 (2009): 1155-1168.

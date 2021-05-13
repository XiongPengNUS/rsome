<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Distributionally robust optimization

## The general formulation for distributionally robust optimization models <a name="section3.1"></a>

The RSOME package supports optimization models that fit the general formulation below

$$
\begin{align}
\min ~~ &\sup\limits_{\mathbb{P}\in\mathcal{F}_0} \mathbb{E}_{\mathbb{P}} \left\{\pmb{a}_0^T(\tilde{s}, \tilde{\pmb{z}})\pmb{x}(\tilde{s}) + \pmb{b}_0^T\pmb{y}(\tilde{s}, \tilde{\pmb{z}}) + c_0(\tilde{s}, \tilde{\pmb{z}})\right\} &&\\
\text{s.t.} ~~ & \mathbb{E}_{\mathbb{P}}\left\{\pmb{a}_m^T(\tilde{s}, \tilde{\pmb{z}})\pmb{x}(\tilde{s}) + \pmb{b}_m^T\pmb{y}(\tilde{s}, \tilde{\pmb{z}}) + c_m(\tilde{s}, \tilde{\pmb{z}})\right\} \leq 0, && \forall \mathbb{P}\in\mathcal{F}_m, \forall m\in \mathcal{M}_1 \\
&\pmb{a}_m^T(s, \pmb{z})\pmb{x}(s) + \pmb{b}_m^T\pmb{y}(s, \pmb{z}) + c_m(s, \pmb{z}) \leq 0, && \forall \pmb{z} \in \mathcal{Z}_{sm}, \forall s \in \mathcal{S}, \forall m\in \mathcal{M}_2 \\
& \pmb{x}(s) \in \mathcal{X}_s && s \in \mathcal{S} \\
& x_i \in \mathcal{A}\left(\mathcal{C}_x^i\right) && \forall i \in \mathcal{I} \\
& y_n \in \overline{\mathcal{A}}\left(\mathcal{C}_y^n, \mathcal{J}_y^n\right) && \forall n \in \mathcal{N}
\end{align}
$$

with \\(s\in\mathcal{S}\\) the indices of discrete scenarios, \\(\pmb{z}\in\mathbb{R}^J\\) an array of random variables. Decision variables \\(\pmb{x}(s)\in\mathbb{R}^I\\), \\(\pmb{y}(s, \pmb{z})\in\mathbb{R}^N\\), and other problem parameters may vary according to different scenarios. The formulation above minimizes (or maximizes) the worst-case expectation of the objective function over an ambiguity set \\(\mathcal{F}_0\\), subject to a number of constraints that consider the worst-case expectations or worst-case realizations.

The adaptation of decision \\(x_i(\cdot)\\) is captured by an event-wise static rule \\(\mathcal{A}\left(\mathcal{C}\right)\\), expressed as

$$
\mathcal{A}\left(\mathcal{C}\right) = \left\{
x: |\mathcal{S}| \mapsto \mathbb{R} \left|
\begin{array}
~x(s)=x^{\mathcal{E}}, \mathcal{E}=\mathcal{H}_{\mathcal{C}}(s) \\
\text{for some } x^{\mathcal{E}} \in \mathbb{R}
\end{array}
\right.
\right\},
$$

where \\(\mathcal{C}\\) is a collection of mutually exclusive and collectively exhaustive (MECE) events. Note that if \\(C\\) is a one-event set \\(\\{\mathcal{S}\\}\\), the decision \\(\pmb{x}\\) is non-adaptive because it is fixed for all scenarios.

The adaptation of decision \\(y_n(\cdot)\\) is captured by an event-wise affine rule \\(\overline{\mathcal{A}}\left(\mathcal{C}, \mathcal{J}\right)\\), expressed as

$$
\overline{\mathcal{A}}\left(\mathcal{C}, \mathcal{J}\right) = \left\{
y: |\mathcal{S}|\times\mathbb{R}^{|\mathcal{J}|} \mapsto \mathbb{R}
\left|
\begin{array}
~y(s, \pmb{z}) = y^0(s) + \sum\limits_{j\in\mathcal{J}}y_j^z(s)z_j \\
\text{for some }y^0, y_j^z \in \mathcal{A}\left(\mathcal{C}\right), j\in\mathcal{J}
\end{array}
\right.
\right\},
$$

where \\(\mathcal{J}\\) is a subset of all random variables, so the decision \\(y_n(\cdot)\\) is affinely adaptive to random variables specified by \\(\mathcal{J}\\).

The event-wise ambiguity set \\(\mathcal{F}_m\\) is defined as

$$
\begin{align}
\mathcal{F}_m = \left\{
\mathbb{P}\in\mathcal{P}_0\left(\mathbb{R}^{J}\times|\mathcal{S}|\right)
\left|
\begin{array}
~\left(\tilde{\pmb{z}}, \tilde{s}\right) \sim \mathbb{P} & \\
\mathbb{E}_{\mathbb{P}}[\tilde{\pmb{z}}|\tilde{s}\in\mathcal{E}_{km}] \in \mathcal{Q}_{km} & \forall k \in [K] \\
\mathbb{P}[\tilde{\pmb{z}}\in \mathcal{Z}_{sm}| \tilde{s}=s]=1 & \forall s \in \mathcal{S} \\
\mathbb{P}[\tilde{s}=s] = p_s & \forall s \in \mathcal{S} \\
\text{for some } \pmb{p} \in \mathcal{P}_m &
\end{array}
\right.
\right\},~~~
\forall m \in \mathcal{M}_1
\end{align},
$$

for given events \\(\mathcal{E}\_{km}\\) and closed and convex sets \\(\mathcal{Q}\_{km}\\), \\(\mathcal{Z}_{sm}\\), and \\(\mathcal{P}_m\subseteq\\{\pmb{p}\in \mathbb{R}\_{\+\+}^{\|\mathcal{S}\|}\|\sum\_{s\in\mathcal{S}}p_s=1\\}\\). The interpretation of each line of the ambiguity set is:

1. The array \\(\tilde{\pmb{z}}\\) of random variables and an random senario index \\(\tilde{s}\\) follows the distribution \\(\mathcal{P}\\).
2. The conditional expectation of \\(\tilde{\pmb{z}}\\) under an event \\(\mathcal{E}\_{km}\\) is defined by a closed and convex set \\(\mathcal{Q}_{km}\\).
3. The support of \\(\tilde{\pmb{z}}\\) in each scenario \\(s\\) is defined by a closed and convex set \\(\mathcal{Z}_{sm}\\).
3. The probability of each scenario is denoted by a probability variable \\(p_s\\).
4. The array \\(\pmb{p}\\) of all probability variables is defined by the set \\(\mathcal{P}_m\\).

In the remaining part of the guide, we will introduce the RSOME code for for specifying the event-wise recourse adaptation and the ambiguity set.

## Introduction to the `rsome.dro` environment <a name="section3.2"></a>

In general, the `rsome.dro` modeling environment is very similar to `rsome.ro` discussed in the section [Introduction to the <code>rsome.ro</code> environment](get_start#section1.2), so almost all array operations, indexing and slicing syntax could be applied to `dro` models. The unique features of the `dro` model mainly come from the scenario-representation of uncertainties and a different way of specifying the event-wise adaptation of decision variables.

### Models
Similar to the `rsome.ro` modeling environment, the `dro` models are all defined upon `Model` type objects, which are created by the constructor `Model()` imported from the sub-packaage `rsome.dro`.

```
class Model(builtins.object)
 |  Model(scens=1, name=None)
 |  
 |  Returns a model object with the given number of scenarios.
 |  
 |  Parameters
 |  ----------
 |  scens : int or array-like objects
 |      The number of scenarios, if it is an integer. It could also be
 |      an array of scenario indices.
 |  name : str
 |      Name of the model
 |  
 |  Returns
 |  -------
 |  model : rsome.dro.Model
 |      A model object
```

It can be seen that the `dro` models are different from the `ro` models in the first argument `scens`, which specifies the scenario-representation of uncertainties. The `scens` argument can be specified as an integer, indicating the number of scenarios, or an array of scenario indices, as shown by the following sample code.

```python
from rsome import dro

model1 = dro.Model(5)       # A DRO model with 5 scenarios

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model2 = dro.Model(labels)  # A DRO model with 5 scenarios: 'sunny', 'cloudy', ...
```

### Decision variables

In the general formulation, decision variables \\(\pmb{x}(s)\\) are event-wise static and \\(\pmb{y}(s, \pmb{z})\\) are event-wise affinely adaptive. The `dro` modeling environment does not differentiate the two in creating these variables. Both types of decision variables are created by the `dvar()` method of the model object, with the same syntax as the `rsome.ro` models.

```
dvar(shape=(1,), vtype='C', name=None) method of rsome.dro.Model instance
    Returns an array of decision variables with the given shape
    and variable type.

    Parameters
    ----------
    shape : int or tuple
        Shape of the variable array.
    vtype : {'C', 'B', 'I'}
        Type of the decision variables. 'C' means continuous; 'B'
        means binary, and 'I" means integer.
    name : str
        Name of the variable array

    Returns
    -------
    new_var : rsome.lp.DecVar
        An array of new decision variables
```

All decision variables are created to be non-adaptive at first. The event-wise and affine adaptation can be then specified by the method `adapt()` of the variable object. Details of specifying the adaptive decisions are provided in the section [Event-wise recourse adaptations](#section3.3)

### Random variables
The syntax of creating random variables for a `dro` model is exactly the same as the `ro` models. You may refer to the section [Random variables and uncertainty sets](ro_rsome#section2.2) for more details.

The `dro` model supports the expectation of random variables, so that we could define the expectation sets \\(\mathcal{Q}_{km}\\) in the ambiguity set \\(\mathcal{F}_m\\). This is different from `ro` models. The expectation is indicated by the `E()` function imported from `rsome`, as demonstrated by the sample code below.

```python
from rsome import dro
from rsome import E     # Import E as the expectation operator

model = dro.Model()
z = model.rvar((3, 5))  # z is a 3x5 random variable

E(z) <= 1               # E(z) is smaller than or equal to zero
E(z) >= -1              # E(z) is larger than or equal to -1
E(z).sum() == 0         # Sum of E(z) is zero
```

## Event-wise ambiguity set <a name="section3.3"></a>

### Create an ambiguity set

Ambiguity sets \\(\mathcal{F}_m\\) of a `dro` model can be created by the `ambiguity()` method. The associated scenario indices of the ambiguity set can be accessed by the `s` attribute of the ambiguity set object, as shown by the following sample code.

```python
from rsome import dro

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)     # Create a model with 5 scenarios
fset = model.ambiguity()      # Create an ambiguity set of the model
print(fset)                   # Print scenarios of the model (ambiguity set)
```

```
Scenario indices:
sunny     0
cloudy    1
rainy     2
windy     3
snowy     4
dtype: int64
```
In the example above, strings in the list `lables` become the labels of the scenario indices. If an integer instead of an array is used to specify the `scens` argument of the `Model()` constructor, then the labels will be the same as the integer values. Similar to the `pandas.Series` data structure, labels of the scenario indices could be any hashable data types.

RSOME supports indexing and slicing of the scenarios via either the labels or the integer-positions, as shown by the following code segments.

```python
print(fset[2])          # The third scenario
```

```
Scenario index:
2
```

```python
print(fset['rainy'])    # The third scenario
```

```
Scenario index:
2
```

```python
print(fset[:3])         # The first three scenarios
```

```
Scenario indices:
sunny     0
cloudy    1
rainy     2
dtype: int64
```

```python
print(fset[:'rainy'])   # The first three scenarios
```

```
Scenario indices:
sunny     0
cloudy    1
rainy     2
dtype: int64
```

RSOME is also consistent with the `pandas.Sereis` data type in using the `iloc` and `loc` indexers for accessing label-based or integer-position based indices.


```python
print(fset.iloc[:2])            # The first two scenarios via the iloc indexer
```

```
Scenario indices:
sunny     0
cloudy    1
dtype: int64
```

```python
print(fset.loc[:'cloudy'])      # The first two scenarios via the loc indexer
```

```
Scenario indices:
sunny     0
cloudy    1
dtype: int64
```

The indices of the scenarios are crucial in defining components of the ambiguity set, such as sets \\(\mathcal{Q}\_{km}\\), \\(\mathcal{Z}_{sm}\\), and \\(\mathcal{P}_m\\), which will be discussed next.

### \\(\mathcal{Q}_{km}\\) as the support of conditional expectations

According to the formulation of the ambiguity set \\(\mathcal{F}_m\\) presented in the section [The general formulation for distributionally robust optimization models](#section3.1), \\(\mathcal{Q}\_{km}\\) is a closed convex set defined as the support of the conditional expectation of random variables under event \\(k\\), where the event is a collection of scenarios. In the RSOME package, such a collection of scenarios can be specified by the indexing or slicing of the ambiguity set object, and constraints of the \\(\mathcal{Q}\_{km}\\) are defined by the `exptset()` method. For example, the conditional expectations of the random variable \\(\tilde{\pmb{z}}\in\mathbb{R}^3\\) are defined as follows

$$
\begin{align}
&\mathbb{E}\left[\tilde{\pmb{z}}|s\in\mathcal{E}_{1}\right] \in \left\{\pmb{z}: \mathbb{R}^3 \left|
\begin{array}
~|\pmb{z}| \leq 1 \\
\|\pmb{z}\|_1 \leq 1.5
\end{array}
\right.
\right\}, && \mathcal{E}_1 = \{\text{sunny}, \text{cloudy}, \text{rainy}, \text{windy}, \text{snowy}\} \\
&\mathbb{E}\left[\tilde{\pmb{z}}|s\in\mathcal{E}_{2}\right] \in \left\{\pmb{z}: \mathbb{R}^3 \left|
\begin{array}
~\pmb{z} = 0
\end{array}
\right.
\right\}, && \mathcal{E}_2 = \{\text{sunny}, \text{rainy}, \text{snowy}\},
\end{align}
$$

where the first event \\(\mathcal{E}_1\\) is a collection of all scenarios, and the second event \\(\mathcal{E}_2\\) includes scenarios "sunny", "rainy", and "snowy". The supports of the conditional expectations can be specified by the code below.

```python
from rsome import dro
from rsome import E
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()
fset.exptset(abs(E(z)) <= 1,
             norm(E(z), 1) <= 1.5)   # The 1st support of conditional expectations
fset.loc[::2].exptset(E(z) == 0)     # The 2nd support of conditional expectations
```

The ambiguity set `fset` itself represents the event \\(\mathcal{E}_1\\) of all scenarios, and the `loc` indexer is used to form the event \\(\mathcal{E}_2\\) with three scenarios to be included. Besides `loc`, other indexing and slicing expressions described in the previous section can also be used to construct the events for the support sets of the expectations.

### \\(\mathcal{Z}_{sm}\\) as the support of random variables

The support \\(\mathcal{Z}_{sm}\\) of random variables can be specified by the method `suppset()` method, and the scenario information of the support can also be specified by the indexing and slicing expressions of the ambiguity set object. Take the following supports of random variables \\(\tilde{\pmb{z}}\in\mathbb{R}^3\\) for example,

$$
\begin{align}
&\mathbb{P}\left[\tilde{\pmb{z}}\in \left\{
  \left.
  \pmb{z}: \mathbb{R}^3 \left|
  \begin{array}
  ~|\pmb{z}| \leq 1 \\
  \|\pmb{z}\| \leq 1.5
  \end{array}
  \right.
\right\}
\right|
\tilde{s}=s
\right]=1, &\forall s \in \{\text{sunny}, \text{rainy},  \text{snowy}\} \\
&\mathbb{P}\left[\tilde{\pmb{z}}\in \left\{
  \left.
  \pmb{z}: \mathbb{R}^3 \left|
  \begin{array}
  ~\sum\limits_{j=1}^3z_j = 0
  \end{array}
  \right.
\right\}
\right|
\tilde{s}=s
\right]=1, &\forall s \in \{\text{cloudy}, \text{windy}\},
\end{align}
$$

the RSOME code for specifying the supports can be written as follows.

```python
from rsome import dro
from rsome import E
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()
fset.iloc[::2].suppset(abs(z) <= 1,
                       norm(z, 1) <= 1.5)  # The support of z in scenarios 0, 2, 4
fset.iloc[1::2].suppset(z.sum() == 0)      # The support of z in scenarios 1, 3
```

Please note that a valid ambiguity set must have the support sets for all scenarios to be specified. An error message will be given in solving the model if any of the supports are unspecified. RSOME provides a method called `showevents()` to display the specified supports for random variables and their expectations in a data frame, in order to help users check their ambiguity set.

```python
from rsome import dro
from rsome import E
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()
fset.iloc[::2].suppset(abs(z) <= 1,
                       norm(z, 1) <= 1.5)  
fset.iloc[1::2].suppset(z.sum() == 0)      
fset.exptset(abs(E(z)) <= 1,
             norm(E(z), 1) <= 1.5)   
fset.loc[::2].exptset(E(z) == 0)     

fset.showevents()            # Display how the ambiguity set is specified
```

<div>
<table border="1" class="dataframe mystyle">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>expectation 0</th>
      <th>expectation 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sunny</th>
      <td>defined</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>cloudy</th>
      <td>defined</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>rainy</th>
      <td>defined</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>windy</th>
      <td>defined</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>snowy</th>
      <td>defined</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

As the example above shows, supports for all scenarios have been defined, and there are two supports defined for the conditional expectations.

### \\(\mathcal{P}_m\\) as the support of scenario probabilities

In the event-wise ambiguity set, the support of scenario probabilities can also be specified via the calling the method `probset()`, as demonstrated by the following sample code.

```python
from rsome import dro
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()
p = model.p                         # p is the array of scenario probabilities
fset.probset(norm(p-0.2) <= 0.05)   # Define the support of the array p
```

The scenario probabilities are formatted as an array, which can be accessed via the attribute `p` of the model object. Please note that two constraints for probabilities: \\(p_s \geq 0\\), \\(\forall s \in \mathcal{S}\\), and \\(\sum_{s\in\mathcal{S}}p_s=1\\), are already integrated in the ambiguity set, so there is no need to specify them in defining the probability support.

### Special cases

The `rsome.dro` model supports a rather generic event-wise ambiguity set where several different types of support sets are specified separately. In order to make the code more concise and readable, we also provide tools for specifying a few special cases of the generic event-wise ambiguity set.

One of the special cases is the uncertainty set used in conventional robust optimization models. It can be considered as a one-scenario ambiguity set with only the support to be specified. Such an uncertainty set can be specified by the `bou()` method, which meaning "budget of uncertainty", of the model object. For example, the sample code below defines a conventional uncertainty set.

```python
from rsome import dro
from rsome import norm

model = dro.Model()                     # This is a one-scenario model
z = model.rvar(3)

fset = model.bou(abs(z) <= 1,
                 norm(z, 1) <= 1.5)     # Define an uncertainty set
```

Another special case is the simplified version of the WKS ambiguity set (see [Distributionally robust convex optimization](http://www.optimization-online.org/DB_FILE/2013/02/3757.pdf)) introduced in [Adaptive distributionally robust optimization](http://www.optimization-online.org/DB_FILE/2016/03/5353.pdf). This is again a one-scenario ambiguity set. It can be defined by the `wks()` method of the model object, where constraints of both the random variables and their expectations can be specified altogether, as demonstrated by the following sample code.

```python
from rsome import dro
from rsome import norm

model = dro.Model()                         # This is a one-scenario model
z = model.rvar(3)
u = model.rvar(3)

fset = model.wks(abs(z) <= u, u <= 1,
                 norm(z, 1) <= 1.5,
                 E(z) == 0, E(u) == 0.5)    # Define a WKS ambiguity set
```

### The worst-case expectations

Once the ambiguity sets \\(\mathcal{F}_m\\) are defined, they can be used to specify the worst-case expectations in the objective function or constraints. The ambiguity set of the objective function can be specified by the `minsup()` method, for minimizing, or the `maxinf()` method, for maximizing the worst-case expectation of the objective function. The ambiguity set of constraints can be specified by the `forall()`, which is similar to `ro` models in specifying the uncertainty sets. The sample code below is used to illustrate the worst-case expectations.

```python
from rsome import dro
from rsome import norm

model = dro.Model()                    # This is a one-scenario model
z = model.rvar(3)
u = model.rvar(3)
x = model.dvar(3)

fset = model.wks(abs(z) <= u, u <= 1,
                 norm(z, 1) <= 1.5,
                 E(z) == 0,
                 E(u) == 0.5)          # Define a WKS ambiguity set

model.minsup(E((x*z).sum()), fset)     # Minimizing the worst-case exp. over fset
model.maxinf(E((x*z).sum()), fset)     # Minimizing the worst-case exp. over fset

model.st((E(x - z) <= 0).forall(fset)) # The worst-case exp. constraints over fset
```

Similar to the `ro` models, if the ambiguity set is unspecified for a constraint, then by default, its ambiguity set of this constriant is the same as the objective function, so the sample code above is equivalent to the code below.

```python
from rsome import dro
from rsome import norm

model = dro.Model()                    # This is a one-scenario model
z = model.rvar(3)
u = model.rvar(3)
x = model.dvar(3)

fset = model.wks(abs(z) <= u, u <= 1,
                 norm(z, 1) <= 1.5,
                 E(z) == 0,
                 E(u) == 0.5)          # Define a WKS ambiguity set

model.minsup(E((x*z).sum()), fset)     # Minimizing the worst-case exp. over fset
model.maxinf(E((x*z).sum()), fset)     # Minimizing the worst-case exp. over fset

model.st(E(x - z) <= 0)                # The worst-case exp. constraints over fset
```


## Event-wise recourse adaptations <a name="section3.4"></a>

This section introduces how to specify the event-wise static adaptation \\(\mathcal{A}(\mathcal{C})\\) and the event-wise affine adaptation \\(\overline{\mathcal{A}}(\mathcal{C}, \mathcal{J})\\). As we mentioned in prior sections, a decision variable is created to be non-adaptive, in the sense that the event set \\(\mathcal{C} = \\{\mathcal{S}\\}\\) and the dependent set \\(\mathcal{J}=\varnothing\\). These two sets can be modified by the `adapt()` method of the decision variable object, as demonstrated by the following sample code.

```python
from rsome import dro

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)
x = model.dvar(3)

x.adapt('sunny')        # A new event that includes the scenario sunny
x.adapt([2, 3])         # A new event that includes scenarios rainy and windy

x[:1].adapt(z[:2])      # x[:1] is affinely adaptive to z[:2]
```

In the code segment above, the set \\(\mathcal{C}\\) is specified as \\(\\{\\{\text{cloudy}, \text{snowy}\\}, \\{\text{sunny}\\}, \\{\text{rainy}, \text{windy}\\}\\}\\), and the decision variable `x[0]` is affinely adaptive to random variables `z[:2]`.

Similar to the linear decision rule defined in the `ro` modeling environment, coefficients of an event-wise recourse adaptation could be accessed by the `get()` method, where
- `y.get()` returns the constant coefficients of the recourse adaptation. In cases of multiple scenarios, the returned object is a `pandas.Series` with the length to be the same as the number of scenarios. Each element of the series is an array that has the same shape as `y`.
- `y.get(z)` returns the linear coefficients of the recourse adaptation. In cases of multiple scenarios, the returned object is a `pandas.Series` with the length to be the same as the number of scenarios. Each element of the series is an array, and the shape of the array is `z.shape + y.shape`, i.e., the combination of dimensions of `z` and `y`.

## Application examples <a name="section3.5"></a>
### [Multi-stage stochastic financial planning](example_dro_finpl)
### [Distributionally robust optimization for medical appointment scheduling](example_dro_mas)
### [A multi-item newsvendor problem considering the Wasserstein ambiguity set](example_dro_nv)
### [Distributionally robust optimization approaches for a lot-size problem ](example_dro_ls)

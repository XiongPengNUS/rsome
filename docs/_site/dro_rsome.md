<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# RSOME for Distributionally Robust Optimization

## The General Formulation for Distributionally Robust Optimization Models <a name="section3.1"></a>

The RSOME package supports optimization models that fit the general formulation below

$$
\begin{align}
\min ~~ &\sup\limits_{\mathbb{P}\in\mathcal{F}_0} \mathbb{E}_{\mathbb{P}} \left\{\pmb{a}_0^{\top}(\tilde{s}, \tilde{\pmb{z}})\pmb{x}(\tilde{s}) + \pmb{b}_0^{\top}\pmb{y}(\tilde{s}, \tilde{\pmb{z}}) + c_0(\tilde{s}, \tilde{\pmb{z}})\right\} &&\\
\text{s.t.} ~~ &\max\limits_{s\in[S], \pmb{z}\in\mathcal{Z}_{ms}}\left\{\pmb{a}_m^{\top}(s, \pmb{z})\pmb{x}(s) + \pmb{b}_m^{\top}\pmb{y}(s, \pmb{z}) + c_m(s, \pmb{z})\right\} \leq 0, && \forall m\in \mathcal{M}_1 \\
& \sup\limits_{\mathbb{P}\in\mathcal{F}_m}\mathbb{E}_{\mathbb{P}}\left\{\pmb{a}_m^{\top}(\tilde{s}, \tilde{\pmb{z}})\pmb{x}(\tilde{s}) + \pmb{b}_m^{\top}\pmb{y}(\tilde{s}, \tilde{\pmb{z}}) + c_m(\tilde{s}, \tilde{\pmb{z}})\right\} \leq 0, && \forall m\in \mathcal{M}_2 \\
& x_i \in \mathcal{A}\left(\mathcal{C}_x^i\right) && \forall i \in [I_x] \\
& y_n \in \overline{\mathcal{A}}\left(\mathcal{C}_y^n, \mathcal{J}_y^n\right) && \forall n \in [I_y] \\
& \pmb{x}(s) \in \mathcal{X}_s && s \in [S]. \\
\end{align}
$$

Here, parameters of proper dimensions,

$$
\begin{align}
&\pmb{a}_m(s,\pmb{z}) := \pmb{a}_{ms}^0 + \sum\limits_{j\in[J]}\pmb{a}_{ms}^jz_j \\
&\pmb{b}_m(s) := \pmb{b}_{ms}\\
&c_m(\pmb{z}) := c_{ms}^0 + \sum\limits_{j\in[J]}c_{ms}^jz_j
\end{align}
$$

are defined similarly as in the case of [RSOME for robust optimization](ro_rsome#section2.1) and \\(\mathcal{X}_s\\) is an second-order conic (SOC) or exponential conic (EC)representable feasible set of a decision \\(\pmb{x}(s)\\) in the scenario \\(x\\). Constraints indexed by \\(m\in\mathcal{M}_1\\) are satisfied under the worst-case realization, just like those introduced in [RSOME for robust optimization](ro_rsome#section2.1). Another set of constraints indexed by \\(m\in\mathcal{M}_2\\) are satisfied with regard to the worst-case expectation overall all possible distributions defined by an ambiguity set \\(\mathcal{F}_m\\) in the general form introduced in [Chen et al. (2020)](#ref2):

$$
\begin{align}
\mathcal{F}_m = \left\{
\mathbb{P}\in\mathcal{P}_0\left(\mathbb{R}^{J}\times[S]\right)
\left|
\begin{array}
~\left(\tilde{\pmb{z}}, \tilde{s}\right) \sim \mathbb{P} & \\
\mathbb{E}_{\mathbb{P}}[\tilde{\pmb{z}}|\tilde{s}\in\mathcal{E}_{km}] \in \mathcal{Q}_{km} & \forall k \in [K] \\
\mathbb{P}[\tilde{\pmb{z}}\in \mathcal{Z}_{sm}| \tilde{s}=s]=1 & \forall s \in [S] \\
\mathbb{P}[\tilde{s}=s] = p_s & \forall s \in [S] \\
\text{for some } \pmb{p} \in \mathcal{P}_m &
\end{array}
\right.
\right\},~~~
\forall m \in \mathcal{M}_2
\end{align}.
$$

Here for each constraint indexed by \\(m\in\mathcal{M}\_2\\),

1. The conditional expectation of \\(\tilde{\pmb{z}}\\) over events (defined as subsets of scenarios and denoted by \\(\mathcal{E}\_{km}\\) are known to reside in an SOC or EC representable set \\(\mathcal{Q}\_{km}\\);
2. The support of \\(\tilde{\pmb{z}}\\) in each scenario \\(s\in[S]\\) is specified to be another SOC or EC representable set \\(\mathcal{Z}\_{sm}\\);
3. Probabilities of scenarios, collectively denoted by a vector \\(\pmb{p}\\), are constrained by a third SOC or EC representable subset \\(\mathcal{P}_m\subseteq\left\\{\pmb{p}\in \mathbb{R}\_{+\+}^S \left\| \pmb{e}^{\top}\pmb{p}=1 \right\. \right\\}\\) in the probability simplex.

Dynamics of decision-making is captured by the event-wise recourse adaptation for wait-and-see decisions of two types—-the <i>event-wise static adaptation</i> denoted by \\(\mathcal{A}(C)\\) as well as the <i>event-wise affine adaptation</i> denoted by \\(\overline{\mathcal{A}}(\mathcal{C}, \mathcal{J})\\). In particular, given a fixed number of \\(S\\) scenarios and a
partition \\(\mathcal{C}\\) of these scenarios (<i>i.e.</i>, a collection of mutually exclusive and collectively exhaustive
events), the event-wise recourse adaptation is formally defined as follows:

$$
\begin{align}
&\mathcal{A}\left(\mathcal{C}\right) = \left\{
x: [S] \mapsto \mathbb{R} \left|
\begin{array}
~x(s)=x^{\mathcal{E}}, \mathcal{E}=\mathcal{H}_{\mathcal{C}}(s) \\
\text{for some } x^{\mathcal{E}} \in \mathbb{R}
\end{array}
\right.
\right\}, \\
&\overline{\mathcal{A}}\left(\mathcal{C}, \mathcal{J}\right) = \left\{
y: [S]\times\mathbb{R}^{J} \mapsto \mathbb{R}
\left|
\begin{array}
~y(s, \pmb{z}) = y^0(s) + \sum\limits_{j\in\mathcal{J}}y^j(s)z_j \\
\text{for some }y^0(s), y^j(s) \in \mathcal{A}\left(\mathcal{C}\right), j\in\mathcal{J}
\end{array}
\right.
\right\}
\end{align}.
$$

Here, \\(\mathcal{H}\_{\mathcal{C}}: [S] \mapsto \mathcal{C}\\) is a function such that \\(\mathcal{H}_{\mathcal{C}}= \mathcal{E}\\) maps the scenario \\(s\\) to the only event \\(\mathcal{E}\\) in \\(\mathcal{C}\\) that contains \\(s\\), and \\(\mathcal{J} \subseteq [J]\\) is an index subset of random components \\(\tilde{z}_1\\),..., \\(\tilde{z}_J\\) that the affine adaptation depends on. In the remaining part of the guide, we will introduce the RSOME code for specifying the event-wise ambiguity set and recourse adaptation rules.

## Introduction to the `rsome.dro` Environment <a name="section3.2"></a>

In general, the `rsome.dro` modeling environment is very similar to `rsome.ro` discussed in the section [Introduction to the <code>rsome.ro</code> Environment](get_start#section1.2), so almost all array operations, indexing and slicing syntax could be applied to `dro` models. The unique features of the `dro` model mainly come from the scenario-representation of uncertainties and a different way of specifying the event-wise adaptation of decision variables.

### Models
Similar to the `rsome.ro` modeling environment, the `dro` models are all defined upon `Model` type objects, which are created by the constructor `Model()` imported from the sub-package `rsome.dro`.

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

model1 = dro.Model(5)       # a DRO model with 5 scenarios

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model2 = dro.Model(labels)  # a DRO model with 5 scenarios given by a list.
```

### Decision Variables

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

### Random Variables
The syntax of creating random variables for a `dro` model is exactly the same as the `ro` models. You may refer to the section [Random variables and uncertainty sets](ro_rsome#section2.2) for more details.

The `dro` model supports the expectation of random variables, so that we could define the expectation sets \\(\mathcal{Q}_{km}\\) in the ambiguity set \\(\mathcal{F}_m\\). This is different from `ro` models. The expectation is indicated by the `E()` function imported from `rsome`, as demonstrated by the sample code below.

```python
from rsome import dro
from rsome import E     # import E as the expectation operator

model = dro.Model()
z = model.rvar((3, 5))  # 3x5 random variables as a 2D array

E(z) <= 1               # E(z) is smaller than or equal to zero
E(z) >= -1              # E(z) is larger than or equal to -1
E(z).sum() == 0         # sum of E(z) is zero
```

## Event-wise Ambiguity Set <a name="section3.3"></a>

### Create an Ambiguity Set

Ambiguity sets \\(\mathcal{F}_m\\) of a `dro` model can be created by the `ambiguity()` method. The associated scenario indices of the ambiguity set can be accessed by the `s` attribute of the ambiguity set object, as shown by the following sample code.

```python
from rsome import dro

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)     # create a model with 5 scenarios

fset = model.ambiguity()      # create an ambiguity set of the model
print(fset)                   # print scenarios of the model (ambiguity set)
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
In the example above, strings in the list `labels` become the labels of the scenario indices. If an integer instead of an array is used to specify the `scens` argument of the `Model()` constructor, then the labels will be the same as the integer values. Similar to the `pandas.Series` data structure, labels of the scenario indices could be any hashable data types.

RSOME supports indexing and slicing of the scenarios via either the labels or the integer-positions, as shown by the following code segments.

```python
print(fset[2])          # the third scenario
```

```
Scenario index:
2
```

```python
print(fset['rainy'])    # the third scenario
```

```
Scenario index:
2
```

```python
print(fset[:3])         # the first three scenarios
```

```
Scenario indices:
sunny     0
cloudy    1
rainy     2
dtype: int64
```

```python
print(fset[:'rainy'])   # the first three scenarios
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
print(fset.iloc[:2])            # the first two scenarios via the iloc[] indexer
```

```
Scenario indices:
sunny     0
cloudy    1
dtype: int64
```

```python
print(fset.loc[:'cloudy'])      # the first two scenarios via the loc[] indexer
```

```
Scenario indices:
sunny     0
cloudy    1
dtype: int64
```

The indices of the scenarios are crucial in defining components of the ambiguity set, such as sets \\(\mathcal{Q}\_{km}\\), and \\(\mathcal{Z}_{sm}\\), which will be discussed next.

### \\(\mathcal{Q}_{km}\\) as the Support of Conditional Expectations

According to the formulation of the ambiguity set \\(\mathcal{F}_m\\) presented in the section [The general formulation for distributionally robust optimization models](#section3.1), the SOC or EC representable set \\(\mathcal{Q}\_{km}\\) is defined as the support of the conditional expectation of random variables under the event \\(\mathcal{E}_k\\), which is a collection of selected scenarios. In the RSOME package, such a collection of scenarios can be specified by the indexing or slicing of the ambiguity set object, and constraints of the \\(\mathcal{Q}\_{km}\\) are defined by the `exptset()` method of the ambiguity set object. The help information of the `exptset()` method is given below.

```
exptset(*args) method of rsome.dro.Ambiguity instance
    Specify the uncertainty set of the expected values of random
    variables for an ambiguity set.

    Parameters
    ----------
    args : tuple
        Constraints or collections of constraints as iterable type of
        objects, used for defining the feasible region of the uncertainty
        set of expectations.

    Notes
    -----
    RSOME leaves the uncertainty set of expectations unspecified if the
    input argument is an empty iterable object.
```

Take the supports of conditional expectations below, for example,

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

where the first event \\(\mathcal{E}_1\\) is a collection of all scenarios, and the second event \\(\mathcal{E}_2\\) includes scenarios "sunny", "rainy", and "snowy". The conditional expectation information of the ambiguity set can be specified by the code below.

```python
from rsome import dro
from rsome import E
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()             # create an ambiguity set
fset.exptset(abs(E(z)) <= 1,
             norm(E(z), 1) <= 1.5)   # the 1st support of conditional expectations
fset.loc[::2].exptset(E(z) == 0)     # the 2nd support of conditional expectations
```

The ambiguity set `fset` itself represents the event \\(\mathcal{E}_1\\) of all scenarios, and the `loc` indexer is used to form the event \\(\mathcal{E}_2\\) with three scenarios included. Besides `loc`, other indexing and slicing expressions described in the previous section can also be used to construct the events for the support sets of the expectations.

### \\(\mathcal{Z}_{sm}\\) as the Support of Random Variables

The support \\(\mathcal{Z}_{sm}\\) of random variables can be specified by the method `suppset()` method, and the scenario information of the support can also be specified by the indexing and slicing expressions of the ambiguity set object. The help information of the `suppset()` method is given below.

```
suppset(*args) method of rsome.dro.Ambiguity instance
    Specify the support set(s) of an ambiguity set.

    Parameters
    ----------
    args : Constraints or iterables
        Constraints or collections of constraints as iterable type of
        objects, used for defining the feasible region of the support set.

    Notes
    -----
    RSOME leaves the support set unspecified if the given argument is
    an empty iterable object.
```

Take the following supports of random variables \\(\tilde{\pmb{z}}\in\mathbb{R}^3\\) for example,

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
                       norm(z, 1) <= 1.5)  # the support of z in scenarios 0, 2, 4
fset.iloc[1::2].suppset(z.sum() == 0)      # the support of z in scenarios 1, 3
```

Note that a valid ambiguity set must have the support sets for all scenarios to be specified. An error message will be given in solving the model if any of the supports are unspecified. RSOME provides a method called `showevents()` to display the specified supports for random variables and their expectations in a data frame, in order to help users check their ambiguity set.

```python
from rsome import dro
from rsome import E
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()
fset.iloc[::2].suppset(abs(z) <= 1, norm(z, 1) <= 1.5)  
fset.iloc[1::2].suppset(z.sum() == 0)      
fset.exptset(abs(E(z)) <= 1, norm(E(z), 1) <= 1.5)   
fset.loc[::2].exptset(E(z) == 0)     

fset.showevents()            # display how the ambiguity set is specified
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

### \\(\mathcal{P}_m\\) as the Support of Scenario Probabilities

In the event-wise ambiguity set, the support of scenario probabilities can also be specified via calling the method `probset()`. The help information of this method is given below.

```
probset(*args) method of rsome.dro.Ambiguity instance
    Specify the uncertainty set of the scenario probabilities for an
    ambiguity set.

    Parameters
    ----------
    args : tuple
        Constraints or collections of constraints as iterable type of
        objects, used for defining the feasible region of the uncertainty
        set of scenario probabilities.

    Notes
    -----
    RSOME leaves the uncertainty set of probabilities unspecified if the
    input argument is an empty iterable object.
```

The following sample code specifies an ellipsoidal uncertainty set of the scenario probabilities.

```python
from rsome import dro
from rsome import norm

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)

fset = model.ambiguity()
p = model.p                         # p is the array of scenario probabilities
fset.probset(norm(p-0.2) <= 0.05)   # define the support of the array p
```

The scenario probabilities are formatted as a one-dimensional array, which can be accessed via the attribute `p` of the model object. Notice that two underlying constraints for probabilities: \\(\pmb{p}\geq \pmb{0}\\) and \\(\pmb{e}^{\top}\pmb{p}=1\\), are already integrated in the ambiguity set, so there is no need to specify them in defining the support of scenario probabilities.

## Event-wise Recourse Adaptations <a name="section3.4"></a>

Here we introduce how the event-wise static adaptation \\(\mathcal{A}(\mathcal{C})\\) and the event-wise affine adaptation \\(\overline{\mathcal{A}}(\mathcal{C}, \mathcal{J})\\) are specified in RSOME. Note that decision variables created in the `rsome.dro` modeling environment are initially non-adaptive, in the sense that the event set \\(\mathcal{C} = \\{[S]\\}\\) and the dependent set \\(\mathcal{J}=\varnothing\\). These two sets can be modified by the `adapt()` method of the decision variable object for specifying the decision adaptation, as demonstrated by the following sample code.

```python
from rsome import dro

labels = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy']
model = dro.Model(labels)
z = model.rvar(3)
x = model.dvar(3)

x.adapt('sunny')        # x adapts to the sunny
x.adapt([2, 3])         # x adapts to the event of rainy and windy

x[:2].adapt(z[:2])      # x[:2] is affinely adaptive to z[:2]
```

In the code segment above, the scenario partition \\(\mathcal{C}\\) is specified to have three events: \\(\\{\text{sunny}\\}\\), \\(\\{\text{rainy}, \text{windy}\\}\\), and collectively the remaining scenarios \\(\\{\text{cloudy}, \text{snowy}\\}\\). The decision variable `x[:2]` is set to be affinely adaptive to the random variable `z[:2]`.

Similar to the linear decision rule defined in the `ro` module, coefficients of an event-wise recourse adaptation could be accessed by the `get()` method after solving the model, where
- `y.get()` returns the constant coefficients of the recourse adaptation. In cases of multiple scenarios, the returned object is a `pandas.Series` with the length to be the same as the number of scenarios. Each element of the series is an array that has the same shape as `y`.
- `y.get(z)` returns the linear coefficients of the recourse adaptation. In cases of multiple scenarios, the returned object is a `pandas.Series` with the length to be the same as the number of scenarios. Each element of the series is an array, and the shape of the array is `y.shape + z.shape`, <i>i.e.</i>, the combination of dimensions of `y` and `z`.

## The Worst-Case Expectations <a name="section3.5"></a>

Once the ambiguity sets \\(\mathcal{F}_m\\) and the recourse adaptation of decision variables are defined, the worst-case expectations in the objective function or constraints can be then specified. The ambiguity set of the objective function can be specified by the `minsup()` method, for minimizing, or the `maxinf()` method, for maximizing, the objective function involving the worst-case expectations. The documentation of the `minsup()` method is given below.

```
minsup(obj, ambset) method of rsome.dro.Model instance
    Minimize the worst-case expected objective value over the given
    ambiguity set.

    Parameters
    ----------
    obj
        Objective function involving random variables
    ambset : Ambiguity
        The ambiguity set defined for the worst-case expectation

    Notes
    -----
    The ambiguity set defined for the objective function is considered
    the default ambiguity set for the distributionally robust model.
```

Similar to `ro` models, ambiguity sets of constraints indexed by \\(m\in\mathcal{M}_2\\), concerning the worst-case expectations, can be specified by the `forall()` method, and if the ambiguity set of such constraints are unspecified, then by default, the underlying ambiguity set is the same as \\(\mathcal{F}_0\\) defined for the objective function. As for the worst-case constraints indexed by \\(m\in\mathcal{M}_1\\), the uncertainty set can also be specified by the `forall()` method, and if the uncertainty set is unspecified, such worst-case constraints are defined upon support sets \\(\mathcal{Z}\_{0s}\\) of the default ambiguity set \\(\mathcal{F}_0\\), for each scenario \\(s\in[S]\\). The sample code below demonstrates how the worst-case expectations can be formulated in the objective function and constraints.


```python
from rsome import dro
from rsome import norm
from rsome import E

model = dro.Model()             # create a model with one scenario
z = model.rvar(3)
u = model.rvar(3)
x = model.dvar(3)

fset = model.ambiguity()        # create an ambiguity set
fset.suppset(abs(z) <= u, u <= 1,
             norm(z, 1) <= 1.5)
fset.exptset(E(z) == 0, E(u) == 0.5)

model.maxinf(E(x @ z), fset)    # maximize the worst-case expectation over fset

model.st((E(x - z) <= 0))       # worst-case expectation over fset
model.st(2x >= z)               # worst-case over the support of fset
```


## Application Examples <a name="section3.6"></a>
### [Distributionally Robust Portfolio](example_dro_portfolio)
### [Distributionally Robust Medical Appointment](example_dro_mas)
### [Multi-Item Newsvendor Problem with Wasserstein Ambiguity Sets](example_dro_nv)
### [Adaptive Distributionally Robust Lot-Sizing](example_dro_ls)
### [Distributionally Robust Vehicle Pre-Allocation](example_dro_vehicle)
### [Multi-Stage Inventory Control](example_dro_inv)
### [Multi-Stage Stochastic Financial Planning](example_dro_finpl)


## Reference

<a id="ref1"></a>

Bertsimas, Dimitris, Melvyn Sim, and Meilin Zhang. 2019. [Adaptive distributionally robust optimization](http://www.optimization-online.org/DB_FILE/2016/03/5353.pdf). <i>Management Science</i> <b>65</b>(2) 604-618.

<a id="ref2"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.

<a id="ref3"></a>

Wiesemann, Wolfram, Daniel Kuhn, Melvyn Sim. 2014. [Distributionally robust convex optimization](https://pubsonline.informs.org/doi/abs/10.1287/opre.2014.1314). <i>Operations Research</i> <b>62</b>(6) 1358–1376.

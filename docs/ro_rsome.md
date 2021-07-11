<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# RSOME for Robust Optimization

## General Formulation for Robust Optimization Models <a name="section2.1"></a>

The `rsome.ro` module in RSOME is designed for robust optimization problems, where tailored modeling tools are developed for specifying random variables, uncertainty sets, and objective functions or constraints under worst-case scenarios that may arise from the uncertainty set. Let \\(\pmb{z}\in\mathbb{R}^J\\) be a vector of random variables and \\(\pmb{x}\in\mathbb{R}^{I_x}\\)(resp., \\(\pmb{y}(\pmb{z})\in\mathbb{R}^{I_y}\\)) be the here-and-now  (resp., non-anticipative wait-and-see) decision made before (resp., after) the uncertainty realizes. Models supported in the `ro` module can be cast into the following general format:

$$
\begin{align}
\min ~ &\max\limits_{\pmb{z}\in\mathcal{Z}_0} ~\left\{\pmb{a}_0^{\top}(\pmb{z})\pmb{x} + \pmb{b}_0^{\top}\pmb{y}(\pmb{z}) + c_0(\pmb{z})\right\} &&\\
\text{s.t.} ~ & \max\limits_{\pmb{z}\in\mathcal{Z}_M}\left\{\pmb{a}_m^{\top}(\pmb{z})\pmb{x} + \pmb{b}_m^{\top}\pmb{y}(\pmb{z}) + c_m(\pmb{z})\right\} \leq 0, && \forall m \in \mathcal{M}_1 \\
& y_i \in \mathcal{L}(\mathcal{J}_i) &&\forall i \in [I_y] \\
& \pmb{x} \in \mathcal{X}. &&
\end{align}
$$

Here \\(\mathcal{X}\subseteq \mathbb{R}^{I_x}\\) is a feasible of \\(\pmb{x}\\) which is second-order cone representable, \\(\pmb{b}_m\in\mathbb{R}^{I_y}\\), \\(m \in \mathcal{M}_1\cup \\{0\\}\\) are fixed parameters of \\(\pmb{y}\\), and uncertain parameters \\(\pmb{a}_m(\pmb{z})\\) as well as \\(c_m(\pmb{z})\\) are expressed as affine mappings of random variable \\(\pmb{z}\\):

$$
\begin{align}
\pmb{a}_m(\pmb{z}) := \pmb{a}_m^0 + \sum\limits_{j\in[J]}\pmb{a}_m^jz_j ~~\text{and}~~ c_m(\pmb{z}) := c_m^0 + \sum\limits_{j\in[J]}c_m^jz_j.
\end{align}
$$

where \\(\pmb{a}_m^j\in\mathbb{R}^{I_x}\\) and \\(c_m^j\in\mathbb{R}\\), indexed by \\(j\in[J]\cup\\{0\\}\\) and \\(\mathcal{M}_1\cup \\{0\\}\\), are proper coefficients. The wait-and-see decision \\(\pmb{y}\\), which can potentially be an arbitrary functional of uncertainty realization \\(\pmb{z}\\), is infinite-dimensional, and thus is hard to optimize. A common robust optimization technique for tractability, called linear decision rule (or affine decision rule), is to restrict y to simpler and easy-to-optimize affine functions in the following form:

$$
\begin{align}
\mathcal{L}(\mathcal{J}) := \left\{y: \mathbb{R}^{[\mathcal{J}]} \mapsto \mathbb{R} \left|
y(\pmb{z}) = y^0 + \sum\limits_{j\in\mathcal{J}}y^jz_j
\right.
\right\}.
\end{align}
$$

The RSOME package provides rich algebraic tools for specifying random variables arrays, linear decision rules, uncertainty sets, the worst-case objective and constraints of a robust model, which will be introduced in the subsequent sections.

## Random Variables and Uncertainty Sets <a name="section2.2"></a>

Similar to decision variables, random variables are created as arrays in RSOME, and the shapes of random variable arrays are specified by the `rvar()` method of the `Model` object.

```
rvar(shape=(1,), name=None) method of rsome.ro.Model instance
    Returns an array of random variables with the given shape.

    Parameters
    ----------
    shape : int or tuple
        Shape of the variable array.
    name : str
        Name of the variable array

    Returns
    -------
    new_var : rsome.lp.Vars
        An array of new random variables
```

All array operations, convex functions, and NumPy-style syntax for decision variables can also be applied to random variables in defining uncertainty sets. For example, let \\(\pmb{z}\in\mathbb{R}^5\\) be the vector of random variables, an uncertainty set \\(\mathcal{Z}_0 = \left\\{\pmb{z} \left\| \\|\pmb{z}\\|\_{\infty} \leq 1, \\|\pmb{z}\\|\_1 \leq 1.5 \right\. \right\\}\\) can be defined as `z_set0` in the following code segment.

```python
from rsome import ro
from rsome import norm
import numpy as np

model = ro.Model()                 # create a model object

z = model.rvar(5)                  # 5 random variables as an 1D array

z_set0 = (norm(z, np.inf) <= 1,    # the infinity-norm constraint
          norm(z, 1) <= 1.5)       # the one-norm constraint
```

Note that an uncertainty set is a collection of constraints, written as an iterable Python object, such as `tuple` or `list`. These constraints are then used in specifying the worst-case objective function and constraints, which are introduced in the next section.

## The Worst-Case Objective and Constraints <a name="section2.3"></a>

In the case of minimizing or maximizing the worst-case objective function in ROMSE, we may use the `minmax()` or the `maxmin()` method of the `Model` object to specify the objective function and the uncertainty set.

```
minmax(obj, *args) method of rsome.ro.Model instance
    Minimize the maximum objective value over the given uncertainty set.

    Parameters
    ----------
    obj
        Objective function involving random variables
    *args
        Constraints or collections of constraints of random variables
        used for defining the uncertainty set

    Notes
    -----
    The uncertainty set defined for the objective function is considered
    the default uncertainty set for the robust model.

```
The documentation shows that the objective function of the robust model is specified by the first argument `obj`, while the remaining arguments could be constraints, or collection of constraints, used for defining the uncertainty set. The `minmax()` and `maxmin()` methods thus enable two approaches for specifying the worst-case objective, as shown by the sample code below.

```python
from rsome import ro
from rsome import norm
import numpy as np

model = ro.Model()   

x = model.dvar(5)                # 5 decision variables as a 1D array
z = model.rvar(5)                # 5 random variables as a 1D array

# define the uncertainty set z_set0 as a tuple
z_set0 = (norm(z, np.inf) <= 1,   
          norm(z, 1) <= 1.5)

# the worst-case objective over the uncertainty set z_set0
model.minmax(x @ z, z_set0)    

# the worst-case objective over an uncertainty set defined by two constraints
model.minmax(x @ z, norm(z, np.inf) <= 1, norm(z, 2) <= 1.25)
```

Similar to deterministic constraints, the worst-case constraints can be defined using the NumPy-style array operations, and the associated uncertainty set is specified using the `forall()` method of the constraint.

```
forall(*args) method of rsome.lp.RoConstr instance
    Specify the uncertainty set of the constraints involving random
    variables. The given arguments are constraints or collections of
    constraints used for defining the uncertainty set.

    Notes
    -----
    The uncertainty set defined by this method overrides the default
    uncertainty set defined for the worst-case objective.
```

The `forall()` method enables users to flexibly define the worst-case constraints, as demonstrated by the sample code below.

```python
from rsome import ro
from rsome import norm

model = ro.Model()   

x = model.dvar(5)
z = model.rvar(5)

# define an ellipsoidal uncertainty set z_set0
z_set0 = norm(z, 2) <= 1.5

# the worst-case objective over the uncertainty set z_set0
model.minmax(x @ z, z_set0)    

# worst-case constraints over the uncertainty set z_set0
model.st((x * z <= 2).forall(z_set0))      
model.st((x*z + x >= 0).forall(z_set0))        

# worst-case constraints over different uncertainty sets defined by a loop
model.st((x[:i] <= z[:i].sum()).forall(norm(z, 2) <= i*0.5) for i in range(1, 6))
```

Note that if the uncertainty set of a robust constraint is unspecified, then by default, its uncertainty set is \\(\mathcal{Z}_0\\), defined by the `minmax()` or `maxmin()` methods for the worst-case objective. The sample code above is hence equivalent to the following code segment.

```python
from rsome import ro
from rsome import norm

model = ro.Model()   

x = model.dvar(5)
z = model.rvar(5)

# define an ellipsoidal uncertainty set z_set0
z_set0 = norm(z, 2) <= 1.5

# the worst-case objective over the default uncertainty set z_set0
model.minmax(x @ z, z_set0)    

# worst-case constraints over the default uncertainty set z_set0
model.st(x * z <= 2)      
model.st(x*z + x >= 0)        

# worst-case constraints over uncertainty sets different from the default one
model.st((x[:i] <= z[:i].sum()).forall(norm(z, 2) <= i*0.5) for i in range(1, 6))
```

The sample code above shows that the Python version of RSOME is able to specify defferent uncertainty sets \\(\mathcal{Z}_m\\), \\(m\in\mathcal{M}_1\cup\\{0\\}\\), for the objective function (with index 0) and each of the constraints (with index \\(m\in\mathcal{M}_1\\)).  Such a framework is more flexible than that in the MATLAB version introduced in [Chen et al.  (2020)](#ref2) and can be used to address a rich range of robust models, including the distributional interpretation of robust
formulation in [Xu et al. (2012)](#ref4), the notion of Pareto robustly optimal solution discussed in [de Ruiter et al. (2016)](#ref3), as well as the sample robust optimization models proposed by [Bertsimas et al. (2021)](#ref1).

## Linear Decision Rules for Adaptive Decision-Making <a name="section2.4"></a>

The `rsome.ro` modeling environment also supports linear decision rules for non-anticipative decision-making. A linear decision rule object can be created by the `ldr()` method of an `ro` model. Details of the method are provided below.

```
ldr(shape=(1,), name=None) method of rsome.ro.Model instance
    Returns an array with the given shape of linear decision rule
    variables.

    Parameters
    ----------
    shape : int or tuple
        Shape of the variable array.
    name : str
        Name of the variable array

    Returns
    -------
    new_var : rsome.ro.DecRule
        An array of new linear decision rule variables
```

Decision rules are also defined as arrays, as shown by the following examples.

```python
from rsome import ro

model = ro.Model()

x = model.ldr((2, 4))       # 2x4 decision rule variables as a 2D array
y = model.ldr((3, 5, 4))    # 3x5x4 decision rule variables as a 3D array

print(x)
print(y)
```

```
2x4 decision rule variables
3x5x4 decision rule variables
```

As mentioned in previous sections, the decision rule \\(y_i\\) is restricted to being affinely depend on a subset \\(\mathcal{J}_i\\) of random variables, and such a subset can be specified by the `adapt()` method of the decision rule object, as shown by the following code segment.

```python
z = model.rvar((2, 4))      # 2x4 random variables as a 2D array
u = model.rvar(5)           # 5 random variables as a 1D array

x.adapt(z)                  # all elements of x depends on all z elements
y[2, 3:, :].adapt(z[0, 1])  # y[2, 3:, :] depends on z[0, 1]
y[1, 3:, :].adapt(u[3:])    # y[1, 3:, :] depends on u[3:]
```

Once the decision rules are created and the affine dependency on random variables is specified, the aforementioned array operations and syntax can be applied to decision rule arrays in constructing constraints involving adaptive decisions. The affine dependency must be specified before using decision rule variables in constraints, otherwise an error message will be given.

Finally, after the model is solved, coefficients of a decision rule `y` could be accessed by the `get()` method. More specifically:
- `y.get()` returns the constant coefficients of the decision rule `y`. The returned array has the same shape as the decision rule array `y`.
- `y.get(z)` returns the linear coefficients of `y` with respect to the random variable `z`. The shape of the returned array is `y.shape + z.shape`, <i>i.e.</i>, the combination of dimensions of `y` and `z`. For example, if `c = y.get(z)` where `y.dim=2`, and `z.dim=2`, the returned coefficients are presented as a four-dimensional array `c` and `c[i, j]` gives the linear coefficients of `y[i, j]` with respect to the random variable `z`.

## Application Examples <a name="section2.5"></a>

### [Robust Portfolio](example_ro_portfolio)
### [Conditional Value-at-Risk in Robust Portfolio Management](example_ro_cvar_portfolio)
### [Robust/Robustness Knapsack](example_ro_knapsack)
### [Robust Lot-Sizing](example_ls)
### [Joint Production-Inventory](example_ro_inv)

## Reference

<a id="ref1"></a>

Bertsimas, Dimitris, Shimrit Shtern, and Bradley Sturt. 2021. [Two-stage sample robust optimization](https://pubsonline.informs.org/doi/abs/10.1287/opre.2020.2096). <i>Operations Research</i>.

<a id="ref2"></a>

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329–3339.

<a id="ref3"></a>
de Ruiter, Frans JCT, Ruud CM Brekelmans, and Dick den Hertog. 2016. [The impact of the existence of multiple adjustable robust solutions](https://link.springer.com/article/10.1007/s10107-016-0978-6). <i>Mathematical Programming</i> <b>160</b>(1) 531-545.

<a id="ref4"></a>
Xu, Huan, Constantine Caramanis, and Shie Mannor. 2012. [A distributional interpretation of robust optimization](https://pubsonline.informs.org/doi/abs/10.1287/moor.1110.0531). <i>Mathematics of Operations Research</i> <b>37</b>(1) 95-110.

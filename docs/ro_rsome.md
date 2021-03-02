<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Robust optimization

## General formulation for robust optimization models <a name="section2.1"></a>

The `rsome.ro` modeling environment is capable of formulating the robust optimization problems as

$$
\begin{align}
\min ~ &\max\limits_{\pmb{z}\in\mathcal{Z}_0} ~\pmb{a}_0^T(\pmb{z})\pmb{x} + \pmb{b}_0^T\pmb{y}(\pmb{z}) + c_0(\pmb{z}) &&\\
\text{s.t.} ~ & \pmb{a}_1^T(\pmb{z})\pmb{x} + \pmb{b}_1^T\pmb{y}(\pmb{z}) + c_1(\pmb{z}) \leq 0, && \forall \pmb{z}\in\mathcal{Z}_1 \\
& \pmb{a}_2^T(\pmb{z})\pmb{x} + \pmb{b}_2^T\pmb{y}(\pmb{z}) + c_2(\pmb{z}) \leq 0, && \forall \pmb{z}\in\mathcal{Z}_2 \\
& \vdots \\
& \pmb{a}_M^T(\pmb{z})\pmb{x} + \pmb{b}_M^T\pmb{y}(\pmb{z}) + c_M(\pmb{z}) \leq 0, && \forall \pmb{z}\in\mathcal{Z}_M \\
& \pmb{x} \in \mathcal{X} &&
\end{align}
$$

with \\(\pmb{x}\in\mathbb{R}^I\\) an array of decision variables, \\(\pmb{z}\in\mathbb{R}^J\\) an array of random variables, and \\(\pmb{y}(\pmb{z})\\) the linear decision rule that affinely adapts to \\(\pmb{z}\\), which is expressed as

$$
y_n(\pmb{z}) = y_n^0 + \sum\limits_{j\in\mathcal{J}^n} y_{nj}^z z_j, ~~~~n=1, 2, ..., N.
$$

where \\(\mathcal{J}^n\\) is a subset of all random variables that the decision rule \\(y_n(\pmb{z})\\) adapts to. The formulation above suggests that we are minimizing (or maximizing) the objective function under the worst-case realization of the uncertainty set \\(\mathcal{Z}_0\\), subject to constraints under the worst case over uncertainty sets \\(\mathcal{Z}_m\\), \\(m= 1, 2, ..., M\\).

The RSOME package provides rich algebraic tools for specifying random variables arrays, uncertainty sets, the worst-case objective and constraints of the robust model, which will be introduced in the subsequent sections.

## Random variables and uncertainty sets <a name="section2.2"></a>

Random variables of a robust optimization model can be defined by the method <code>rvar()</code> of the model object.

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

Similar to decision variables, random variables are also formulated as arrays, and all array operations and functions aforementioned, including operations between decision and random variables, could be applied.


```python
from rsome import ro

model = ro.Model()          # Create a model object
x = model.dvar((1, 5))      # A 1x5 array of decision varaibles
y = model.dvar((2, 1))      # A 2x1 array of decision variables
z = model.rvar((2, 5))      # A 2x5 array of random variables

model.st(x * z <= 2)        # Multiplication with broadcasting
model.st(y.T@z - x <= 5)    # Matrix multiplication
```

The uncertainty set \\(\mathcal{Z}_0\\) for the objective function can be specified by the method <code>minmax()</code> and <code>maxmin()</code>. Take the following uncertainty set \\(\mathcal{Z}_0\\) for example,  

$$
\begin{align}
\mathcal{Z}_0 = \left\{\pmb{z}:
\|\pmb{z}\|_{\infty} \leq 1,
\|\pmb{z}\|_1 \leq 1.5  
\right\},
\end{align}
$$

it is used to define the worst-case objective functions, which can be written as the following code.

```python
from rsome import ro
import rsome as rso
import numpy as np

model = ro.Model()          
x = model.dvar((2, 5))      
z = model.rvar((1, 5))    

# Define uncertainty set Z0 as a tuple
z_set0 = (rso.norm(z, np.inf) <= 1,   
          rso.norm(z, 1) <= 1.5)

# Minimize the worst-case objective over the uncertainty set Z0
model.minmax((x*z).sum(), z_set0)    

# Maximize the worst-case objective over an uncertainty defined by two constraints
model.maxmin((x*z).sum(),
             rso.norm(z, np.inf) <= 1,
             rso.norm(z, 2) <= 1.25)
```

In the functions <code>minmax()</code> and <code>maxmin()</code>, the first argument is the objective function, and all the remaining arguments are used to specify the constraints of the uncertainty set \\(\mathcal{Z}_0\\). Constraints of the uncertainty set can be provided in an iterable data object, such as a `tuple` or `list`. Alternatively, these constraints can be given as other arguments of the `minmax()` or `maxmin()` methods, as shown by the examples above.


For constraints of the robust model, uncertainty sets can be specified by the <code>forall()</code> method of constraints involving random variables, as shown by the following example.


```python
# Define uncertainty set Z1 as a tuple
z_set1 = (rso.norm(z, np.inf) <= 1.5,
          rso.norm(z, 1) <= 2)

# The constraints over the uncertainty set Z1
model.st((x*z + z >= 0).forall(z_set1))
```

Please note that if the uncertainty set of a robust constraint is not defined, then by default, its uncertainty set is \\(\mathcal{Z}_0\\), defined by the `minmax()` or `maxmin()` methods for the objective. The code below demonstrates a case where one uncertainty set \\(\mathcal{Z}_0\\) applies to the objective function and all constraints.


```python
from rsome import ro
import rsome as rso

model = ro.Model()          
x = model.dvar((2, 5))      
z = model.rvar((1, 5))    

# Define uncertainty set Z0 as a tuple
z_set0 = (rso.norm(z, np.inf) <= 1,   
          rso.norm(z, 1) <= 1.5)

# Define objective function and the uncertainty set
model.minmax((x*z).sum(), z_set0)  

# The uncertainty set Z0 applies to all constraints below
model.st(x*z + z >= 0)
model.st(x*z + x >= 0)
model.st(x >= z)
```

It can be seen that uncertainty sets of the robust model can be flexibly specified. More application examples are presented in the next section.

## Linear decision rules for adaptive decision-making <a name="section2.3"></a>

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

x = model.ldr((2, 4))       # Decision rule variable array x
y = model.ldr((3, 5))       # Decision rule variable array y

print(x)
print(y)
```

```
2x4 decision rule variables
3x5 decision rule variables
```

As mentioned in previous sections, the decision rule \\(y_n(\pmb{z})\\) may affinely depend on a subset \\(\mathcal{J}^n\\) of random variables, and such a subset can be specified by the `adapt()`, as shown by the following code segment.

```python
z = model.rvar((2, 4))      # Random variable array z
u = model.rvar(5)           # Random variable array u

x.adapt(z)                  # All elements of x depends on all z elements
y[2, 3:].adapt(z[0, 1])     # y[2, 3:] depends on z[0, 1]
y[1, 3:].adapt(u[3:])       # y[1, 3:] depends on u[3:]
```

Once the decision rules are created and the affine dependency on random variables is specified, the aforementioned array operations and syntax can be applied to decision rule arrays in constructing constraints involving adaptive decisions. The affine dependency must be specified before using decision rule variables in constraints, otherwise an error message will be given.

Please also note that RSOME does not allow redefinition of the same affine dependency relation, such as the following code segment

```python
y[2, 3:].adapt(z[0, 1])     # y[2, 3:] depends on z[0, 1]
y[:3, :].adapt(z[0])        # y[:3, :] depends on z[0]
```

would give an error message because the affine dependency of decision rules `y[2, 3:]` on the random variable `z[0]` is defined twice.

Finally, after the model is solved, coefficients of a decision rule `y` could be accessed by the `get()` method. More specifically:
- `y.get()` returns the constant coefficients of the decision rule `y`. The returned array has the same shape as the decision rule array `y`.
- `y.get(z)` returns the linear coefficients of `y` with respect to the random variable `z`. The shape of the returned array is `z.shape + y.shape`, i.e., the combination of dimensions of `z` and `y`. For example, if `c = y.get(z)` where `z.dim=2`, and `y.dim=2`, the returned coefficients are presented as a four-dimensional array `c` and `c[i, j]` gives the linear coefficients of `y` with respect to the random variable `z[i, j]`.

## Application examples <a name="section2.4"></a>

### [Robust portfolio optimization](example_ro_portfolio)
### [Conditional value-at-risk with application to robust portfolio management](example_ro_cvar_portfolio)
### [The robust and robustness knapsack problems](example_ro_knapsack)
### [Adaptive robust optimization for a lot-sizing problem](example_ls)
### [The robust production-inventory model](example_ro_inv)

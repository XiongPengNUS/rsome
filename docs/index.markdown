---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home

---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# ROAD: Robust Optimization for Array Data

ROAD (Robust Optimization for Array Data) is a open-source Python package for operations research and generic optimization modeling. ROAD models are constructed by variables, expressions, and constraints formatted as N-dimensional arrays, which are consistent with the NumPy library in syntax and operations, such as indexing and slicing, element-wise operations, broadcasting, and matrix calculation rules. It thus provides a convenient and highly readable way in developing optimization models and applications.

The current version of ROAD supports deterministic linear/second-order cone programs and robust optimization problems. An interface with Gurobi solver is also integrated for the solution of optimization models. Distributionally robust optimization modeling tools based on the [robust stochastic optimization (RSO) framework](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603?af=R) is now under development. Other solver interfaces will be included in the future.

## Introduction

### Installing ROAD and solvers

The ROAD package can be installed with the <code>pip</code> command:

***

<code><b>pip install road</b></code>

***

For the current version, the Gurobi solve is also needed for solving the optimization model, and you may follow [these steps](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html) to complete the solver installation.

### The Dao of ROAD

The ROAD package is largely inspired by [ROME](https://robustopt.com/), the very first software toolbox for robust optimization. We also learned many hard lessons in developing the MATLAB package [RSOME](https://www.rsomerso.com/), hence the "Dao of ROAD", which can be imported from the ROAD package.

See if it works: \\(\sigma_1\\)

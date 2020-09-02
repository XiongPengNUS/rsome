<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Introduction

ROAD (Robust Optimization for Array Data) is an open-source Python package for operations research and generic optimization modeling. ROAD models are constructed by variables, expressions, and constraints formatted as N-dimensional arrays, which are consistent with the NumPy library in syntax and operations, such as indexing and slicing, element-wise operations, broadcasting, and matrix calculation rules. It thus provides a convenient and highly readable way in developing optimization models and applications.

The current version of ROAD supports deterministic linear/second-order cone programs and robust optimization problems. An interface with the Gurobi solver is also integrated for the solution of optimization models. Distributionally robust optimization modeling tools based on the [robust stochastic optimization (RSO) framework](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603?af=R) is now under development. Other solver interfaces will be included in the future.

### Installing ROAD and solvers

The ROAD package can be installed with the <code>pip</code> command:

***

**`pip install road`**

***

For the current version, the Gurobi solve is also needed for solving the optimization model, and you may follow [these steps](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html) to complete the solver installation.

### The Dao of ROAD

The ROAD package is largely inspired by [ROME](https://robustopt.com/), the very first software toolbox for robust optimization. We also learned many hard lessons in developing the MATLAB package [RSOME](https://www.rsomerso.com/), hence the "Dao of ROAD", which can be imported from the ROAD package.


```python
from road import dao
```

    The DAO of ROAD:
    ROME was not built in one day.
    All ROADs lead to ROME.
    Matlab is RSOME!
    The ROAD in Python is more than RSOME!


    ROME: https://robustopt.com/
    RSOME: https://www.rsomerso.com
    ROAD: https://github.com/XiongPengNUS/road



\\[\sigma_1\\]

A test: \\(\sigma_1\\)

```python
for item in a_list:
  print(item)
}
```

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/XiongPengNUS/road/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

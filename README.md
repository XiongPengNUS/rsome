<img src="https://github.com/XiongPengNUS/rsome/blob/master/rsologo.png?raw=true" width=100>

# RSOME: Robust Stochastic Optimization Made Easy

[![PyPI](https://img.shields.io/pypi/v/rsome?label=PyPI)](https://pypi.org/project/rsome/)
[![PyPI - downloads](https://img.shields.io/pypi/dm/rsome?label=PyPI%20downloads)](https://pypi.org/project/rsome/)
[![Commit activity](https://img.shields.io/github/commit-activity/m/xiongpengnus/rsome)](https://github.com/XiongPengNUS/rsome/graphs/commit-activity)
[![Last commit](https://img.shields.io/github/last-commit/xiongpengnus/rsome)](https://github.com/XiongPengNUS/rsome/graphs/commit-activity)
[![tests](https://github.com/XiongPengNUS/rsome/actions/workflows/test.yml/badge.svg)](https://github.com/XiongPengNUS/rsome/actions/workflows/test.yml)
[![Docs](https://github.com/XiongPengNUS/rsome/actions/workflows/pages/pages-build-deployment/badge.svg?label=Docs)](https://github.com/XiongPengNUS/rsome/actions/workflows/pages/pages-build-deployment)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
![GitHub closed issues](https://img.shields.io/github/issues-closed/XiongPengNUS/rsome)
![GitHub issues](https://img.shields.io/github/issues-raw/XiongPengNUS/rsome)

- Website: [RSOME for Python](https://xiongpengnus.github.io/rsome/)
- PyPI: [RSOME 1.2.0](https://pypi.org/project/rsome/)

RSOME (Robust Stochastic Optimization Made Easy) is an open-source Python package for generic modeling of optimization problems (subject to uncertainty). Models in RSOME are constructed by variables, constraints, and expressions that are formatted as N-dimensional arrays. These arrays are consistent with the NumPy library in terms of syntax and operations, including broadcasting, indexing, slicing, element-wise operations, and matrix calculation rules, among others. In short, RSOME provides a convenient platform to facilitate developments of robust optimization models and their applications.

## Content

- [Installation](#section2)
- [Solver interfaces](#section3)
- [Getting started](#section4)
- [Team](#section5)
- [Citation](#section6)

## Installation <a id="section2"></a>

The RSOME package can be installed by using the <code>pip</code> command:
***
**`pip install rsome`**
***

### Solver interfaces <a id="section3"></a>

The RSOME package transforms robust or distributionally robust optimization models into deterministic linear or conic programming problems, and solved by external solvers. Details of compatible solvers and their interfaces are presented in the following table.

| Solver | License  type | Required version | RSOME interface | Second-order cone constraints| Exponential cone constraints | Semidefiniteness constraints
|:-------|:--------------|:-----------------|:----------------|:------------------------|:---------------------|:--------------|
|[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)| Open-source | >= 1.9.0 | `lpg_solver` | No | No | No |
|[CyLP](https://github.com/coin-or/cylp)| Open-source | >= 0.9.0 | `clp_solver` | No | No | No |
|[OR-Tools](https://developers.google.com/optimization/install) | Open-source | >= 7.5.7466 | `ort_solver` | No | No | No |
|[ECOS](https://github.com/embotech/ecos-python) | Open-source | >= 2.0.10 | `eco_solver` | Yes | Yes | No |
|[Gurobi](https://www.gurobi.com/documentation/9.0/quickstart_mac/ins_the_anaconda_python_di.html)| Commercial | >= 9.1.0 | `grb_solver` | Yes | No | No |
|[Mosek](https://docs.mosek.com/9.2/pythonapi/install-interface.html) | Commercial | >= 10.0.44 | `msk_solver` | Yes | Yes | Yes |
|[CPLEX](https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) | Commercial | >= 12.9.0.0 | `cpx_solver` | Yes | No | No |
|[COPT](https://www.shanshu.ai/copt) | Commercial | >= 6.5.3 | `cpt_solver` | Yes | No | No |

## Getting started <a id="section4"></a>

Documents of RSOME are provided as follows:
- [RSOME quick start](https://xiongpengnus.github.io/rsome/)
- [RSOME users guide](https://xiongpengnus.github.io/rsome/user_guide)
- [Application examples](https://xiongpengnus.github.io/rsome/examples)

## Team <a id="section5"></a>

RSOME is a software project supported by Singapore Ministry of Education Tier 3 Grant *Science of Prescriptive Analytics*. It is primarly developed and maintained by [Zhi Chen](https://www.cb.cityu.edu.hk/staff/zchen96/), [Melvyn Sim](https://bizfaculty.nus.edu.sg/faculty-details/?profId=127), and [Peng Xiong](https://bizfaculty.nus.edu.sg/faculty-details/?profId=543). Many other researchers, including Erick Delage, Zhaowei Hao, Long He, Zhenyu Hu, Jun Jiang, Brad Sturt, Qinshen Tang, as well as anonymous users and paper reviewers, have helped greatly in the way of developing RSOME.

## Citation <a id="section6">

If you use RSOME in your research, please cite our papers:

- Chen, Zhi, and Peng Xiong. 2023. [RSOME in Python: an open-source package for robust stochastic optimization made easy](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2023.1291). Forthcoming in <i>INFORMS Journal on Computing</i>.

- Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. [Robust stochastic optimization made easy with RSOME](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3603). <i>Management Science</i> <b>66</b>(8) 3329-3339.

Bibtex entry:

```
@article{chen2021rsome,
  title={RSOME in Python: an open-source package for robust stochastic optimization made easy},
  author={Chen, Zhi and Xiong, Peng},
  journal={INFORMS Journal on Computing},
  year={2023},
  month={Mar},
  day={30},
  publisher={INFORMS}
}
```

```
@article{chen2020robust,
  title={Robust stochastic optimization made easy with RSOME},
  author={Chen, Zhi and Sim, Melvyn and Xiong, Peng},
  journal={Management Science},
  volume={66},
  number={8},
  pages={3329--3339},
  year={2020},
  publisher={INFORMS}
}
```

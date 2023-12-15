from .lp import def_sol as solve
import scipy


version = scipy.__version__
name = 'SciPy Optimize'
info = f'{name} {version}'

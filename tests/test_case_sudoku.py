from rsome import ro
from rsome import dro
from rsome import clp_solver as clp
from rsome import cpx_solver as cpx
import numpy as np


def test_ro_model():

    puzzle = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 2],
                       [6, 0, 0, 1, 9, 5, 0, 0, 0],
                       [0, 9, 8, 0, 0, 0, 0, 6, 0],
                       [8, 0, 0, 0, 6, 0, 0, 0, 3],
                       [4, 0, 0, 8, 0, 3, 0, 0, 1],
                       [7, 0, 0, 0, 2, 0, 0, 0, 6],
                       [0, 6, 0, 0, 0, 0, 2, 8, 0],
                       [0, 0, 0, 4, 1, 9, 0, 0, 5],
                       [0, 0, 0, 0, 8, 0, 0, 7, 9]])

    model = ro.Model()
    x = model.dvar((9, 9, 9), vtype='B')

    model.min(0)
    model.st(x.sum(axis=0) == 1,
             x.sum(axis=1) == 1,
             x.sum(axis=2) == 1)
    is_fixed = puzzle > 0
    model.st(x[is_fixed, puzzle[is_fixed]-1] == 1)
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            model.st(x[i: i+3, j: j+3, :].sum(axis=(0, 1)) == 1)

    model.solve(clp)

    solution = np.array([[5., 3., 4., 6., 7., 8., 9., 1., 2.],
                         [6., 7., 2., 1., 9., 5., 3., 4., 8.],
                         [1., 9., 8., 3., 4., 2., 5., 6., 7.],
                         [8., 5., 9., 7., 6., 1., 4., 2., 3.],
                         [4., 2., 6., 8., 5., 3., 7., 9., 1.],
                         [7., 1., 3., 9., 2., 4., 8., 5., 6.],
                         [9., 6., 1., 5., 3., 7., 2., 8., 4.],
                         [2., 8., 7., 4., 1., 9., 6., 3., 5.],
                         [3., 4., 5., 2., 8., 6., 1., 7., 9.]])

    assert model.get() == 0
    mip_sol = (x.get() * np.arange(1, 10).reshape((1, 1, 9))).sum(axis=2)
    assert (mip_sol.round() == solution).all()


def test_dro_model():

    puzzle = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 2],
                       [6, 0, 0, 1, 9, 5, 0, 0, 0],
                       [0, 9, 8, 0, 0, 0, 0, 6, 0],
                       [8, 0, 0, 0, 6, 0, 0, 0, 3],
                       [4, 0, 0, 8, 0, 3, 0, 0, 1],
                       [7, 0, 0, 0, 2, 0, 0, 0, 6],
                       [0, 6, 0, 0, 0, 0, 2, 8, 0],
                       [0, 0, 0, 4, 1, 9, 0, 0, 5],
                       [0, 0, 0, 0, 8, 0, 0, 7, 9]])

    model = dro.Model()
    x = model.dvar((9, 9, 9), vtype='B')

    model.min(0)
    model.st(x.sum(axis=0) == 1,
             x.sum(axis=1) == 1,
             x.sum(axis=2) == 1)
    is_fixed = puzzle > 0
    model.st(x[is_fixed, puzzle[is_fixed]-1] == 1)
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            model.st(x[i: i+3, j: j+3, :].sum(axis=(0, 1)) == 1)

    model.solve(cpx)

    solution = np.array([[5., 3., 4., 6., 7., 8., 9., 1., 2.],
                         [6., 7., 2., 1., 9., 5., 3., 4., 8.],
                         [1., 9., 8., 3., 4., 2., 5., 6., 7.],
                         [8., 5., 9., 7., 6., 1., 4., 2., 3.],
                         [4., 2., 6., 8., 5., 3., 7., 9., 1.],
                         [7., 1., 3., 9., 2., 4., 8., 5., 6.],
                         [9., 6., 1., 5., 3., 7., 2., 8., 4.],
                         [2., 8., 7., 4., 1., 9., 6., 3., 5.],
                         [3., 4., 5., 2., 8., 6., 1., 7., 9.]])

    assert model.get() == 0
    mip_sol = (x.get() * np.arange(1, 10).reshape((1, 1, 9))).sum(axis=2)
    assert (mip_sol.round() == solution).all()

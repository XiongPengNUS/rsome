<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Integer programming for Sudoku

In this section we will use a [Sudoku](https://en.wikipedia.org/wiki/Sudoku) game to illustrate how to use integer and multi-dimensional arrays in RSOME. Sudoku is a popular number puzzle. The goal is to place the digits in \[1,9\] on a nine-by-nine grid, with some of the digits already filled in. The solution must satisfy the following four rules:

1. Each cell contains an integer in \[1,9\].
2. Each row must contain each of the integers in \[1,9\].
3. Each column must contain each of the integers in \[1,9\].
4. Each of the nine 3x3 squares with bold outlines must contain each of the integers in \[1,9\].

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg/1280px-Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg.png" width=200>
</p>

The Sudoku game can be considered as a feasibility optimization problem with the objective to be zero and constraints used to fulfill above rules. Consider a binary variable \\(x_{ijk}\in \\{0, 1\\}\\), with\\(i \in [0, 8]\\), \\(j \in [0, 8]\\), and \\(k \in [0, 8]\\). It equals to one if an integer \\(k+1\\) is placed in a cell at the \\(i\\)th row and \\(j\\)th column. Let \\(a_{ij}\\) be the known number at the \\(i\\)th row and \\(j\\)th column, with \\((i, j)\in\mathcal{I}\times\mathcal{J}\\) as \\(\mathcal{I}\\) and \\(\mathcal{J}\\) are the row and column indices of numbers with known values, the Sudoku game can be thus written as the following integer program:

$$
\begin{align}
\min~&0 \\
\text{s.t.}~& \sum\limits_{i=0}^8x_{ijk} = 1, \forall j \in [0, 8], k \in [0, 8] \\
& \sum\limits_{j=0}^8x_{ijk} = 1, \forall i \in [0, 8], k \in [0, 8] \\
& \sum\limits_{k=0}^8x_{ijk} = 1, \forall i \in [0, 8], j \in [0, 8] \\
& x_{ij(a_{ij}-1)} = 1, \forall i \in \mathcal{I}, j \in \mathcal{J} \\
& \sum\limits_{m=0}^2\sum\limits_{n=0}^2x_{(i+m), (j+m), k} = 1, \forall i \in \{0, 3, 6\}, j \in \{0, 3, 6\}, k \in [0, 8]
\end{align}
$$

In the following code, we are using RSOME to implement such a model.


```python
import rsome as rso
import numpy as np
from rsome import ro
from rsome import grb_solver as grb

# Sudoku puzzle
# zeros represent unknown numbers
puzzle = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 2],
                   [6, 0, 0, 1, 9, 5, 0, 0, 0],
                   [0, 9, 8, 0, 0, 0, 0, 6, 0],
                   [8, 0, 0, 0, 6, 0, 0, 0, 3],
                   [4, 0, 0, 8, 0, 3, 0, 0, 1],
                   [7, 0, 0, 0, 2, 0, 0, 0, 6],
                   [0, 6, 0, 0, 0, 0, 2, 8, 0],
                   [0, 0, 0, 4, 1, 9, 0, 0, 5],
                   [0, 0, 0, 0, 8, 0, 0, 7, 9]])

# create model and binary decision variables
model = ro.Model()
x = model.dvar((9, 9, 9), vtype='B')

# objective is set to be zero
model.min(0 * x.sum())

# constraints 1 to 3
model.st(x.sum(axis=0) == 1,
         x.sum(axis=1) == 1,
         x.sum(axis=2) == 1)

# constraints 4
i, j = np.where(puzzle)
model.st(x[i, j, puzzle[i, j]-1] == 1)

# constraints 5
for i in range(0, 9, 3):
    for j in range(0, 9, 3):
        model.st(x[i: i+3, j: j+3, :].sum(axis=(0, 1)) == 1)

# solve the integer programming problem
model.solve(grb)
```

    Being solved by Gurobi...
    Solution status: 2
    Running time: 0.0017s


The binary variable \\(x_{ijk}\\) is defined to be a three-dimensional array <code>x</code> with the shape to be <code>(9, 9, 9)</code>. Note that in RSOME, the objective function cannot be specified as a numeric constant, we then use the expression <code>0 * x.sum()</code> as the objective. Based on the decision variable <code>x</code>, each set of constraints can be formulated as the array form by using the <code>sum</code> method. The method <code>sum()</code> in RSOME is consistent with that in NumPy, where you may use the <code>axis</code> argument to specify along which axis the sum is performed.

The Sudoku problem and the its solution are presented below.


```python
print(puzzle)                   # display the Sudoku puzzle
```

    [[5 3 0 0 7 0 0 0 2]
     [6 0 0 1 9 5 0 0 0]
     [0 9 8 0 0 0 0 6 0]
     [8 0 0 0 6 0 0 0 3]
     [4 0 0 8 0 3 0 0 1]
     [7 0 0 0 2 0 0 0 6]
     [0 6 0 0 0 0 2 8 0]
     [0 0 0 4 1 9 0 0 5]
     [0 0 0 0 8 0 0 7 9]]



```python
x_sol = x.get().astype('int')   # retrieve the solution as integers

print((x_sol * np.arange(1, 10).reshape((1, 1, 9))).sum(axis=2))
```

    [[5 3 4 6 7 8 9 1 2]
     [6 7 2 1 9 5 3 4 8]
     [1 9 8 3 4 2 5 6 7]
     [8 5 9 7 6 1 4 2 3]
     [4 2 6 8 5 3 7 9 1]
     [7 1 3 9 2 4 8 5 6]
     [9 6 1 5 3 7 2 8 4]
     [2 8 7 4 1 9 6 3 5]
     [3 4 5 2 8 6 1 7 9]]

Note that in defining "constraints 4", variables `i` and `j` represent the row and column indices of the fixed elements, which can be retrieved by the `np.where()` function. An alternative approach is to use the boolean indexing of arrays, as the code below.

```python
# an alternative approach for constraints 4
is_fixed = puzzle > 0
model.st(x[is_fixed, puzzle[is_fixed]-1] == 1)
```

The variable `is_fixed` is an array with elements to be `True` if the numbers are fixed and `False` if the numbers are unknown. Such an array with boolean data can also be used as indices, thus defining the same constraints.

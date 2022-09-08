import rsome as rso
from rsome import dro
from rsome import grb_solver as grb
from rsome import eco_solver as eco
from rsome import E
import numpy as np
import numpy.random as rd
import pytest


@pytest.mark.parametrize('array, const', [
    (rd.rand(3, 7), rd.rand()),
    (rd.rand(4, 7), rd.rand()),
    (rd.rand(5, 1), rd.rand())
])
def test_norm_one(array, const):

    target = abs(array).sum(axis=1) + const
    ns = array.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])

    z = m.rvar(array.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)

    expr = rso.norm(x, 1) + const
    m.minsup(E(a), fset)
    m.st(a >= expr)
    m.st(x == z)
    m.solve(grb)

    assert abs(m.get() - target.mean()) < 1e-4
    for s in range(ns):
        assert abs(a.get()[s] - abs(array[s]).sum() - const) < 1e-4

    target = -abs(array).sum(axis=1) + const
    ns = array.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])

    z = m.rvar(array.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)

    expr = - rso.norm(x, 1) + const
    m.maxinf(E(a), fset)
    m.st(a <= expr)
    m.st(x == z)
    m.solve(grb)

    assert abs(m.get() - target.mean()) < 1e-4
    for s in range(ns):
        assert abs(a.get()[s] + abs(array[s]).sum() - const) < 1e-4

    with pytest.raises(ValueError):
        rso.norm(x, 1) + const >= 0

    with pytest.raises(ValueError):
        -rso.norm(x, 1) + const <= 0

    with pytest.raises(TypeError):
        rso.norm(x, 1) + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(3, 5), rd.rand()),
    (rd.rand(5, 1), rd.rand())
])
def test_norm_two(array, const):

    target = (array**2).sum(axis=1)**0.5 + const
    ns = array.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])

    z = m.rvar(array.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)

    expr = rso.norm(x, 2) + const
    m.minsup(E(a), fset)
    m.st(a >= expr)
    m.st(x == z)
    m.solve(grb)

    assert abs(m.get() - target.mean()) < 5e-4
    for s in range(ns):
        assert abs(a.get()[s] - (array[s]**2).sum()**0.5 - const) < 1e-4

    target = -(array**2).sum(axis=1)**0.5 + const
    ns = array.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])

    z = m.rvar(array.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)

    expr = - rso.norm(x, 2) + const
    m.maxinf(E(a), fset)
    m.st(a <= expr)
    m.st(x == z)
    m.solve(grb)

    assert abs(m.get() - target.mean()) < 5e-4
    for s in range(ns):
        assert abs(a.get()[s] + (array[s]**2).sum()**0.5 - const) < 5e-4

    with pytest.raises(ValueError):
        rso.norm(x, 2) + const >= 0

    with pytest.raises(ValueError):
        -rso.norm(x, 2) + const <= 0

    with pytest.raises(TypeError):
        rso.norm(x, 2) + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(3, 7), rd.rand()),
    (rd.rand(4, 7), rd.rand()),
    (rd.rand(5, 1), rd.rand())
])
def test_norm_inf(array, const):

    target = array.max(axis=1) + const
    ns = array.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])

    z = m.rvar(array.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)

    expr = rso.norm(x, 'inf') + const
    m.minsup(E(a), fset)
    m.st(a >= expr)
    m.st(x == z)
    m.solve(grb)

    assert abs(m.get() - target.mean()) < 1e-4
    for s in range(ns):
        assert abs(a.get()[s] - (array[s]).max() - const) < 1e-4

    target = -array.max(axis=1) + const
    ns = array.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])

    z = m.rvar(array.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)

    expr = -rso.norm(x, 'inf') + const
    m.maxinf(E(a), fset)
    m.st(a <= expr)
    m.st(x == z)
    m.solve(grb)

    assert abs(m.get() - target.mean()) < 1e-4
    for s in range(ns):
        assert abs(a.get()[s] + (array[s]).max() - const) < 1e-4

    with pytest.raises(ValueError):
        rso.norm(x, 'inf') + const >= 0

    with pytest.raises(ValueError):
        -rso.norm(x, 'inf') + const <= 0

    with pytest.raises(TypeError):
        rso.norm(x, 'inf') + const == 0


@pytest.mark.parametrize('array, const', [
    (rd.rand(2, 3), rd.rand(2, 1)),
    (rd.rand(2, 3), rd.rand(2, 3)),
    (rd.rand(3, 1), rd.rand(3, 1)),
    (rd.rand(2, 1, 2), rd.rand(2, 2, 2)),
    (rd.rand(2, 2, 2), rd.rand(2, 2, 1)),
    (rd.rand(2, 2, 2), rd.rand(2, 1, 2)),
    (rd.rand(2, 1, 2, 1), rd.rand(2, 2, 2, 1))
])
def test_squares(array, const):

    target = array**2 + const
    ns = target.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    d = m.dvar(target.shape[1:])
    x = m.dvar(array.shape[1:])
    y = m.dvar(const.shape[1:])

    z = m.rvar(array.shape[1:])
    u = m.rvar(const.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        d.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = rso.square(x) + y
    m.minsup(E(a), fset)
    m.st(a >= d.sum())
    m.st(d >= expr)
    m.st(x == z, y == u)
    m.solve(grb)

    assert abs(m.get() - target.mean(axis=0).sum()) < 5e-4
    for s in range(ns):
        assert (abs(d.get()[s] - (array**2 + const)[s]) < 5e-4).all()

    target = -array**2 + const
    ns = target.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    d = m.dvar(target.shape[1:])
    x = m.dvar(array.shape[1:])
    y = m.dvar(const.shape[1:])

    z = m.rvar(array.shape[1:])
    u = m.rvar(const.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        d.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = -rso.square(x) + y
    m.maxinf(E(a), fset)
    m.st(a <= d.sum())
    m.st(d <= expr)
    m.st(x == z, y == u)
    m.solve(grb)

    assert abs(m.get() - target.mean(axis=0).sum()) < 5e-4
    for s in range(ns):
        assert (abs(d.get()[s] - (-array**2 + const)[s]) < 5e-4).all()

    with pytest.raises(ValueError):
        rso.square(x) + y >= 0

    with pytest.raises(ValueError):
        -rso.square(x) + y <= 0

    with pytest.raises(TypeError):
        rso.square(x) + y == 0


rd.seed(1)


@pytest.mark.parametrize('array, const', [
    (rd.rand(2, 3), rd.rand(2)),
    (rd.rand(5, 1), rd.rand(5))
])
def test_square_sum(array, const):

    target = (array**2).sum(axis=1) + const
    ns = target.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])
    y = m.dvar(const.shape[1:])

    z = m.rvar(array.shape[1:])
    u = m.rvar(const.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = rso.sumsqr(x) + y
    m.minsup(E(a), fset)
    m.st(a >= expr)
    m.st(x == z, y == u)
    m.solve(grb)

    assert (abs(a.get().values - target) < 5e-4).all()
    assert abs(m.get() - target.mean()) < 5e-4

    target = -(array**2).sum(axis=1) + const
    ns = target.shape[0]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(array.shape[1:])
    y = m.dvar(const.shape[1:])

    z = m.rvar(array.shape[1:])
    u = m.rvar(const.shape[1:])
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = -rso.sumsqr(x) + y
    m.maxinf(E(a), fset)
    m.st(a <= expr)
    m.st(x == z, y == u)
    m.solve(grb)

    assert (abs(a.get().values - target) < 5e-4).all()
    assert abs(m.get() - target.mean()) < 5e-4

    with pytest.raises(ValueError):
        rso.sumsqr(x) + y >= 0

    with pytest.raises(ValueError):
        -rso.sumsqr(x) + y <= 0

    with pytest.raises(TypeError):
        rso.sumsqr(x) + y == 0


rd.seed(2)


@pytest.mark.parametrize('const1, const2', [
    (rd.rand(2), rd.rand()),
    (rd.rand(4), rd.rand()),
    (rd.rand(6), rd.rand())
])
def test_expcone(const1, const2):

    ns = len(const1)

    m = dro.Model(ns)
    x = m.dvar()
    y = m.dvar()

    z = m.rvar()
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == const1[s])
        x.adapt(s)
        y.adapt(s)
    pr = m.p
    fset.probset(pr == 1/ns)

    m.minsup(E(y), fset)
    m.st(x == z)
    m.st(y.expcone(2*x - 0.87, const2))
    m.solve(eco)

    target = const2 * np.exp((2*const1 - 0.87)/const2)

    assert (abs(y.get().values - target) < 5e-4).all()
    assert abs(m.get() - target.mean()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


rd.seed(1)


@pytest.mark.parametrize('array, const', [
    (rd.rand(3, 2), rd.rand(3, 2)),
    (rd.rand(3, 2, 3), rd.rand(3, 3)),
    (rd.rand(4), rd.rand(4))
])
def test_exp(array, const):

    rd.seed(5)
    ns = array.shape[0]

    target = []
    for s in range(ns):
        target.append(np.exp(array[s] - const[s]))
    shape1 = array.shape[1:]
    shape2 = const.shape[1:]
    shape = target[0].shape

    m = dro.Model(ns)
    a = m.dvar(shape)
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = rso.exp(x - y)
    m.minsup(E(a.sum()), fset)
    m.st(a >= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(np.array(list(a.get())) - np.array(target)) < 5e-4).all()
    assert abs(m.get() - np.array(target).mean(axis=0).sum()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    target = []
    for s in range(ns):
        target.append(-np.exp(array[s] + const[s]))
    shape1 = array.shape[1:]
    shape2 = const.shape[1:]
    shape = target[0].shape

    m = dro.Model(ns)
    a = m.dvar(shape)
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = - rso.exp(x + y)
    m.maxinf(E(a.sum()), fset)
    m.st(a <= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(np.array(list(a.get())) - np.array(target)) < 5e-4).all()
    assert abs(m.get() - np.array(target).mean(axis=0).sum()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


rd.seed(1)


@pytest.mark.parametrize('array, scales', [
    (rd.rand(3, 2), np.maximum(0.2, rd.rand(3, 2))),
    (rd.rand(3, 2, 3), np.maximum(0.2, rd.rand(3, 3))),
    (rd.rand(4), np.maximum(0.2, rd.rand(4)))
])
def test_pexp(array, scales):

    ns = array.shape[0]

    target = []
    for s in range(ns):
        target.append(scales[s]*np.exp(array[s]/scales[s]))
    shape1 = array.shape[1:]
    shape2 = scales.shape[1:]
    shape = target[0].shape

    m = dro.Model(ns)
    a = m.dvar(shape)
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == scales[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = rso.pexp(x, y)
    m.minsup(E(a.sum()), fset)
    m.st(a >= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(np.array(list(a.get())) - np.array(target)) < 5e-4).all()
    assert abs(m.get() - np.array(target).mean(axis=0).sum()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


rd.seed(1)


@pytest.mark.parametrize('array, scales', [
    (rd.rand(3, 2), np.maximum(0.2, rd.rand(3, 2))),
    (rd.rand(3, 2, 3), np.maximum(0.2, rd.rand(3, 3))),
    (rd.rand(4), np.maximum(0.2, rd.rand(4)))
])
def test_plog(array, scales):

    ns = array.shape[0]

    target = []
    for s in range(ns):
        target.append(scales[s]*np.log(array[s]/scales[s]))
    shape1 = array.shape[1:]
    shape2 = scales.shape[1:]
    shape = target[0].shape

    m = dro.Model(ns)
    a = m.dvar(shape)
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == scales[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = rso.plog(x, y)
    m.maxinf(E(a.sum()), fset)
    m.st(a <= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(np.array(list(a.get())) - np.array(target)) < 5e-4).all()
    assert abs(m.get() - np.array(target).mean(axis=0).sum()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


@pytest.mark.parametrize('array, const', [
    (rd.rand(3, 2), rd.rand(3, 2)),
    (rd.rand(3, 2, 3), rd.rand(3, 3)),
    (rd.rand(4), rd.rand(4))
])
def test_log(array, const):

    rd.seed(5)
    ns = array.shape[0]

    target = []
    for s in range(ns):
        target.append(2*np.log(array[s] + const[s]) - 1.23)
    shape1 = array.shape[1:]
    shape2 = const.shape[1:]
    shape = target[0].shape

    m = dro.Model(ns)
    a = m.dvar(shape)
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = 2*rso.log(x + y) - 1.23
    m.maxinf(E(a.sum()), fset)
    m.st(a <= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(np.array(list(a.get())) - np.array(target)) < 5e-4).all()
    assert abs(m.get() - np.array(target).mean(axis=0).sum()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4

    target = []
    for s in range(ns):
        target.append(- np.log(array[s] - const[s] + 1.23))
    shape1 = array.shape[1:]
    shape2 = const.shape[1:]
    shape = target[0].shape

    m = dro.Model(ns)
    a = m.dvar(shape)
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = - rso.log(x - y + 1.23)
    m.minsup(E(a.sum()), fset)
    m.st(a >= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(np.array(list(a.get())) - np.array(target)) < 5e-4).all()
    assert abs(m.get() - np.array(target).mean(axis=0).sum()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


@pytest.mark.parametrize('array, const', [
    (rd.rand(3, 4), rd.rand(3, 4)),
    (rd.rand(4, 3), rd.rand(4, 1)),
    (rd.rand(5, 2), rd.rand(5, 2))
])
def test_entropy(array, const):

    target = - ((array + const) * np.log(array + const)).sum(axis=1)
    ns = target.shape[0]
    shape1 = array.shape[1:]
    shape2 = const.shape[1:]

    m = dro.Model(ns)
    a = m.dvar()
    x = m.dvar(shape1)
    y = m.dvar(shape2)

    z = m.rvar(shape1)
    u = m.rvar(shape2)
    fset = m.ambiguity()
    for s in range(ns):
        fset[s].suppset(z == array[s], u == const[s])
    p = m.p
    fset.probset(p == 1/ns)

    for s in range(ns):
        a.adapt(s)
        x.adapt(s)
        y.adapt(s)

    expr = rso.entropy(x + y)
    m.maxinf(E(a), fset)
    m.st(a <= expr)
    m.st(x == z, y == u)
    m.solve(eco)

    assert (abs(a.get().values - target) < 5e-4).all()
    assert abs(m.get() - target.mean()) < 5e-4
    primal_obj = m.do_math().solve(eco).objval
    dual_obj = m.do_math(primal=False).solve(eco).objval
    assert abs(primal_obj + dual_obj) < 1e-4


@pytest.mark.parametrize('array, const', [
    (rd.rand(5), rd.rand()),
    (rd.rand(3), rd.rand(2)),
    (rd.rand(1), rd.rand(1))
])
def test_quad(array, const):

    vec = rd.rand(array.shape[0]).round(4)
    qmat = np.diag(vec)

    target = array@qmat@array + const

    m = dro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = x.quad(qmat) + const
    m.min(a)
    m.st(a >= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.max()) < 5e-4
    assert type(expr) == rso.lp.DecConvex
    if target.shape == ():
        assert expr.__repr__() == 'a sum of squares expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} sum of squares expression{suffix}'

    target = -array@qmat@array + const

    m = dro.Model()
    a = m.dvar()
    x = m.dvar(array.shape)

    expr = - x.quad(qmat) + const
    m.max(a)
    m.st(a <= expr)
    m.st(x == array)
    m.solve(grb)

    assert abs(m.get() - target.min()) < 5e-4
    assert type(expr) == rso.lp.DecConvex
    if target.shape == ():
        assert expr.__repr__() == 'a sum of squares expression'
    else:
        shape_str = 'x'.join([str(dim) for dim in target.shape])
        suffix = 's' if target.size > 1 else ''
        assert expr.__repr__() == f'{shape_str} sum of squares expression{suffix}'

    with pytest.raises(ValueError):
        x.quad(qmat) + const >= 0

    with pytest.raises(ValueError):
        -x.quad(qmat) + const <= 0

    with pytest.raises(TypeError):
        rso.quad(x, qmat) + const == 0


def test_convex_err():

    model = dro.Model()
    x = model.dvar(5)
    xx = model.dvar((6, 7))

    with pytest.raises(ValueError):
        rso.norm(xx, 1)

    with pytest.raises(ValueError):
        rso.norm(xx, 2)

    with pytest.raises(ValueError):
        rso.norm(xx, 'inf')

    with pytest.raises(ValueError):
        rso.sumsqr(xx)

    vec = rd.rand(7)
    qmat = vec.reshape((vec.size, 1)) @ vec.reshape((1, vec.size))
    with pytest.raises(ValueError):
        rso.quad(xx, qmat)

    qmat = rd.rand(5, 5) + 0.01
    with pytest.raises(ValueError):
        rso.quad(x, qmat)

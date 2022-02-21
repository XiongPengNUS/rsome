from rsome import dro
from rsome import E
import rsome as rso
import pandas as pd
import pytest


def test_integer_scen():

    ns = 10
    model = dro.Model(ns)
    x = model.dvar(5)
    z = model.rvar((3, 5))
    assert z.__repr__() == '3x5 random variables'
    assert E(z).__repr__() == '3x5 expectation of random variables'

    fset = model.ambiguity()

    string = 'Scenario indices: \n' + pd.Series(range(ns), index=range(ns)).__str__()
    assert fset.__repr__() == string
    assert fset.__str__() == string
    events = fset.showevents()
    df = pd.DataFrame({'support': [False]*ns})
    assert (events == df).all().all()

    fset[1:3].suppset(z == 1)
    fset[5:7].suppset(z == 0.5)
    fset[8].suppset(z == 2.5)
    events = fset.showevents()
    df.loc[1:2, 'support'] = True
    df.loc[5:6, 'support'] = True
    df.loc[8, 'support'] = True
    assert (events == df).all().all()

    fset[3:6].exptset(E(z) == 1.5)
    events = fset.showevents()
    df['expectation 0'] = False
    df.loc[3:5, 'expectation 0'] = True
    assert (events == df).all().all()

    fset[4:7].exptset(rso.norm(E(z[0])) <= 1.5,
                      rso.norm(E(z[1])) <= 1.0,
                      rso.norm(E(z[2])) <= 0.5)
    events = fset.showevents()
    df['expectation 1'] = False
    df.loc[4:6, 'expectation 1'] = True
    assert (events == df).all().all()

    p = model.p
    fset.probset(rso.norm(p - 1/ns) <= 1e-2)
    fset.mix_support()

    with pytest.raises(KeyError):
        fset[12]

    with pytest.raises(ValueError):
        fset[3].suppset(x == 1)

    with pytest.raises(TypeError):
        fset[3].suppset(z)

    with pytest.raises(ValueError):
        fset[3].suppset(E(z) == 1)

    with pytest.raises(ValueError):
        fset[3].exptset(z == 1)


def test_array_scen():

    scens = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    ns = len(scens)
    model = dro.Model(scens)
    x = model.dvar(5)
    z = model.rvar((3, 5))
    assert z.__repr__() == '3x5 random variables'
    assert E(z).__repr__() == '3x5 expectation of random variables'

    fset = model.ambiguity()

    string = 'Scenario indices: \n' + pd.Series(range(ns), index=scens).__str__()
    assert fset.__repr__() == string
    assert fset.__str__() == string
    events = fset.showevents()
    df = pd.DataFrame({'support': [False]*ns}, index=scens)
    assert (events == df).all().all()

    fset[1:3].suppset(z == 1)
    fset.iloc[5:7].suppset(z == 0.5)
    fset.iloc[8].suppset(z == 2.5)
    events = fset.showevents()
    df.loc['Feb':'Mar', 'support'] = True
    df.loc['Jun':'Jul', 'support'] = True
    df.loc['Sep', 'support'] = True
    assert (events == df).all().all()

    fset[3:6].exptset(E(z) == 1.5)
    events = fset.showevents()
    df['expectation 0'] = False
    df.loc['Apr':'Jun', 'expectation 0'] = True
    assert (events == df).all().all()

    fset.loc['May':'Aug'].exptset(rso.norm(E(z[0])) <= 1.5,
                                  rso.norm(E(z[1])) <= 1.0,
                                  rso.norm(E(z[2])) <= 0.5)
    events = fset.showevents()
    df['expectation 1'] = False
    df.loc['May':'Aug', 'expectation 1'] = True
    assert (events == df).all().all()

    p = model.p
    fset.probset(rso.norm(p - 1/ns) <= 1e-2)
    fset.mix_support()

    with pytest.raises(KeyError):
        fset['Nov']

    with pytest.raises(ValueError):
        fset['Jul'].suppset(x == 1)

    with pytest.raises(TypeError):
        fset['Jul'].suppset(z)

    with pytest.raises(ValueError):
        fset['Jun':'Aug'].suppset(E(z) == 1)

    with pytest.raises(ValueError):
        fset['Jun':'Aug'].exptset(z == 1)

    with pytest.raises(TypeError):
        model = dro.Model(3.5)

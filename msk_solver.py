import mosek
import numpy as np
from scipy.sparse import coo_matrix
from .socp import SOCProg
import warnings
import time
from .lp import Solution


def solve(form, display=True, export=False):

    numlc, numvar = form.linear.shape
    if isinstance(form, SOCProg):
        qmat = form.qmat
    else:
        qmat = []

    ind_int = np.where(form.vtype == 'I')[0]
    ind_bin = np.where(form.vtype == 'B')[0]

    if ind_bin.size:
        form.ub[ind_bin] = 1
        form.lb[ind_bin] = 0

    ind_ub = np.where((form.ub != np.inf) & (form.lb == -np.inf))[0]
    ind_lb = np.where((form.lb != -np.inf) & (form.ub == np.inf))[0]
    ind_ra = np.where((form.lb != -np.inf) & (form.ub != np.inf))[0]
    ind_fr = np.where((form.lb == -np.inf) & (form.ub == np.inf))[0]
    ind_eq = np.where(form.sense)[0]
    ind_ineq = np.where(form.sense == 0)[0]

    with mosek.Env() as env:

        with env.Task(0, 0) as task:

            task.appendvars(numvar)
            task.appendcons(numlc)

            if ind_ub.size:
                task.putvarboundlist(ind_ub,
                                     [mosek.boundkey.up] * len(ind_ub),
                                     form.lb[ind_ub], form.ub[ind_ub])

            if ind_lb.size:
                task.putvarboundlist(ind_lb,
                                     [mosek.boundkey.lo] * len(ind_lb),
                                     form.lb[ind_lb], form.ub[ind_lb])

            if ind_ra.size:
                task.putvarboundlist(ind_ra,
                                     [mosek.boundkey.ra] * len(ind_ra),
                                     form.lb[ind_ra], form.ub[ind_ra])
            if ind_fr.size:
                task.putvarboundlist(ind_fr,
                                     [mosek.boundkey.fr] * len(ind_fr),
                                     form.lb[ind_fr], form.ub[ind_fr])

            if ind_int.size:
                task.putvartypelist(ind_int,
                                    [mosek.variabletype.type_int]
                                    * len(ind_int))

            if ind_bin.size:
                task.putvartypelist(ind_bin,
                                    [mosek.variabletype.type_int]
                                    * len(ind_bin))

            task.putcslice(0, numvar, form.obj.flatten())
            task.putobjsense(mosek.objsense.minimize)

            coo = coo_matrix(form.linear)
            task.putaijlist(coo.row, coo.col, coo.data)

            if ind_eq.size:
                task.putconboundlist(ind_eq, [mosek.boundkey.fx] * len(ind_eq),
                                     form.const[ind_eq], form.const[ind_eq])
            if ind_ineq.size:
                task.putconboundlist(ind_ineq,
                                     [mosek.boundkey.up] * len(ind_ineq),
                                     [-np.inf] * len(ind_ineq),
                                     form.const[ind_ineq])

            for cone in qmat:
                task.appendcone(mosek.conetype.quad,
                                0.0, cone)

            if display:
                print('Being solved by Mosek...')

            t0 = time.process_time()
            task.optimize()
            stime = time.process_time() - t0

            soltype = mosek.soltype
            solsta = None
            for stype in [soltype.bas, soltype.itr, soltype.itg]:
                try:
                    solsta = task.getsolsta(stype)
                    if display:
                        print('Solution status: {0}'.format(solsta.__repr__()))
                        print('Running time: {0:0.4f}s'.format(stime))

                    break
                except:
                    pass

            xx = [0.] * numvar
            task.getxx(stype, xx)

            if export:
                task.writedata("out.mps")

            if solsta in [mosek.solsta.optimal, mosek.solsta.integer_optimal]:
                solution = Solution(xx[0], xx, solsta)
            else:
                warnings.warn('No feasible solution can be found.')
                solution = None

            return solution

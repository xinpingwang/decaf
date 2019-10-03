import logging
from scipy import optimize

from decaf.base import Solver, Blob

_FMIN = optimize.fmin_l_bfgs_b


class LBFGSSolver(Solver):
    """
    The LBFGS solver.
    TODO read the flowing link to understand LBFGS
    https://en.wikipedia.org/wiki/Limited-memory_BFGS
    https://stats.stackexchange.com/questions/284712/how-does-the-l-bfgs-work
    """
    def __init__(self, **kwargs):
        """
        The LBFGS solver. Necessary args is:
            lbfgs_args: a dictionary containing the parameters to be passed to lbfgs.
        """
        Solver.__init__(self, **kwargs)
        self._lbfgs_args = self.spec.get('lbfgs_args', {})
        self._param = None
        self._net = None

    def _collect_params(self, realloc=False):
        """
        Collect the network parameters into a long vector
        """
        params_list = self._net.params()
        if self._param is None or realloc:
            total_size = sum(p.data().size for p in params_list)
            dtype = max(p.data().dtype for p in params_list)
            self._param = Blob(shape=total_size, dtype=dtype)
            self._param.init_diff()
        current = 0
        for param in params_list:
            size = param.data().size
            self._param.data()[current: current + size] = param.data().flat
            self._param.diff()[current: current + size] = param.diff().flat
            current += size

    def _distribute_params(self):
        """
        Distribute the parameter to the net
        """
        params_list = self._net.params()
        current = 0
        for param in params_list:
            size = param.data().size
            param.data().flat = self._param.data()[current: current+size]
            current += size

    def obj(self, variable):
        """
        The objective function that wraps around the net.
        """
        self._param.data()[:] = variable
        self._distribute_params()
        loss = self._net.execute()
        self._collect_params()
        return loss, self._param.diff()

    def solve(self, net):
        """
        Solves the net.
        """
        # first, run an execute pass to initialize all the parameters
        self._net = net
        net.execute()
        self._collect_params(True)
        # now, run LBFGS
        result = _FMIN(lambda x: self.obj(x), self._param.data(), args=[self], **self._lbfgs_args)
        # put the optimized result to the net
        self._param.data()[:] = result[0]
        self._distribute_params()
        logging.info('Final function value: {}', result[1])

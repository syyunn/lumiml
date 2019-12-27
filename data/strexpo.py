""" Data sampling from stretched exponential distribution - which represents
widely distributed decay distributions

Variables:
    gamma_eval (np.array): Corresponds to discretized decay rate axis.
        Individual gamma values equidistant on the logarithmic scale as
        following:
                    ==============================================
                    Gamma_k = Gamma_0 * e^ (k * pi / omega)
                    where omega defines the resolution of sampling
                    ==============================================
        the gamma_eval is implemented w/ np.logspace which is (base **
        starts/end) of its kwargs.


"""
import numpy as np

import data.utils as utils
from lumiml.simulator import StretchedExponentialDistribution

time_series = np.arange(-2.5, 1, 0.007)

gamma_eval = np.logspace(start=-2.5, stop=1, num=500, base=10, endpoint=True)

utils.show2d(time_series, gamma_eval)

if __name__ == "__main__":
    pass

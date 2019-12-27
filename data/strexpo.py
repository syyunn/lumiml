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


class Data:
    def __init__(self, start, end, num_points, base):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.time_series = np.arange(start,
                                     end,
                                     (end - start) / num_points)
        self.gamma_eval = np.logspace(start=start,
                                      stop=end,
                                      num=num_points,
                                      base=base)

    def show_gamma_eval(self):
        utils.show2d(self.time_series, self.gamma_eval)


if __name__ == "__main__":
    start = -2.5
    end = 1
    num_points = 500
    base = 10

    data = Data(start=start, end=end, num_points=num_points, base=base)
    data.show_gamma_eval()

    pass

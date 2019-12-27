""" Data sampling from stretched exponential distribution - which represents
widely distributed decay distributions

Terms:
    gamma: the term gamma generally refers to the possible decay rate. This
    decay rate assumes to sampled from the parametric distribution, such as
    KWW(stretched exponential function) or bi-exponential(linear combination
    of two exponential distribution)

Variables:
    gamma_eval (np.array): Corresponds to "discretized" decay rate axis.
        Individual gamma values equidistant on the logarithmic scale as
        following:
                ==============================================
                Gamma_k = Gamma_0 * e^ (k * pi / omega)
                where omega defines the resolution of sampling
                ==============================================
        the gamma_eval is implemented w/ np.logspace which is (base **
        starts/end) of its kwargs.

    time_scale(np.array): Corresponds to


"""
import numpy as np

import data.utils as utils
from lumiml.simulator import Simulator, StretchedExponentialDistribution

from lumiml.base import DeltaBasisFeatures
from lumiml.models import PoissonElasticNet
from lumiml.model_selection import PoissonElasticNetCV


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

    def show_gamma_eval(self, reverse=False):
        if reverse:
            _gamma_eval = 1 / self.gamma_eval
        else:
            _gamma_eval = self.gamma_eval
        utils.show2d(self.time_series, _gamma_eval)


if __name__ == "__main__":
    start = -2.5
    end = 1
    num_points = 500
    base = 10

    data = Data(start=start, end=end, num_points=num_points, base=base)
    # data.show_gamma_eval(reverse=False)

    linear_time_space = np.linspace(start=-30, stop=1000, num=10000)

    # data.show_gamma_eval(reverse=False)

    beta_of_series_expansions_rho_of_kww = 0.5  # beta = 1/2 case simplifies
    # the distribution

    tww_characteristic_relaxation_of_time_constant = 5  # uniquely
    # determined by the material's property

    stretched_exponential_dist = StretchedExponentialDistribution(
        beta_kww=beta_of_series_expansions_rho_of_kww,
        gamma_eval=data.gamma_eval,
        n_sum_terms=200,  # resolution of series expansion. refer to
        # https://en.wikipedia.org/wiki/Stretched_exponential_function#cite_note-8
        tau_kww=tww_characteristic_relaxation_of_time_constant)

    background_mean = 100
    signal_noise_rate = 1e4

    simulator = Simulator(
        distribution=stretched_exponential_dist,
        time_scale=linear_time_space,
        background_mean=background_mean,
        signal_noise_rate=signal_noise_rate)

    simulator.simulate_data()

    delta_basis_features = DeltaBasisFeatures(
        g_min=data.gamma_eval[0],
        g_max=data.gamma_eval[-1],
        omega=2*np.pi,  # resolution of time scale
        with_bias=False,
        fix_low_end=False)

    delta_basis_features.fit() # caclulate gamma space

    _filter = simulator.time_scale >= 0

    t = simulator.time_scale[_filter].copy()  # copy to acquire immutability
    # to prevent the value t changes when its ref val changes

    # y = simulator.data_simulated.simulated[_filter].copy()
    #
    # X = delta_basis_features.fit_transform(t[:, np.newaxis])
    #
    # penet = PoissonElasticNet(
    #     alpha=1e-8,
    #     fix_intercept=True,
    #     intercept_guess=background_mean,
    #     max_iter=1
    # )
    #
    # penet_cv = PoissonElasticNetCV(
    #     estimator=penet,
    #     param_grid={'alpha': np.logspace(-9, -5, 31)},
    #     cv=3,
    #     verbose=1,
    #     n_jobs=2
    # )
    #
    # penet_cv.fit(X, y)
    #
    # print(penet_cv.best_estimator_.coef_)

    pass

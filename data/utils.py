"""
Utils to visualize data
"""

import matplotlib.pyplot as plt


def show2d(time_series, values):
    plt.semilogy(time_series, values, 'darkorange')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from . import util


def plot_lag_autocor(axis, array):
    if array.size <= 300:
        ks = np.arange(array.size)
        acor = util.auto_corr(array)
    else:
        ks = np.arange(300)
        acor = util.auto_corr(array[:300]).flatten()
    axis.bar(ks, acor, width=1)


def plot1d(sample, target=None):
    fig = plt.figure(figsize=(14, 7))

    # time series plot
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.set_title('time series')
    ax1.plot(sample.data)
    ax1.grid(True)

    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax2.set_title('distribution')
    # guess good binning
    bins = util.fd_bins(sample)[0]

    ax2.hist(sample.data, bins=bins)
    if target is not None:
        minx = np.min(sample.data)
        maxx = np.max(sample.data)
        x = np.linspace(min(0, minx), max(1, maxx), 1000)
        ax2.plot(x, sample.size * (maxx - minx) / bins * target.pdf(x),
                 label='target distribution')
        ax2.legend()

    ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
    ax3.set_title('lag autocor.')
    plot_lag_autocor(ax3, sample.data.flatten())

    fig.tight_layout()
    return fig


def plot2d(sample, target=None):
    # rough guess for good binning
    fig = plt.figure(figsize=(14, 7))

    # width = 2 if target is None else 3
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.set_title('time series')
    ax1.plot(sample.data[:, 0], label="x")
    ax1.plot(sample.data[:, 1], label="y")
    ax1.grid(True)
    ax1.legend()

    # guess a good binning
    bins = util.fd_bins(sample)

    ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)
    ax2.set_title('distribution')
    counts, xedges, yedges, im = ax2.hist2d(*sample.data.transpose(), bins=bins)
    if target is not None:
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        x = np.linspace(extent[0], extent[1], max(bins)*10)
        y = np.linspace(extent[2], extent[3], max(bins)*10)
        mgrid = np.meshgrid(x, y)
        ax2.contour(x, y, target(*mgrid))
    plt.colorbar(im, ax=ax2)
    ax2.set_aspect('equal')

    # plot
    ax3 = plt.subplot2grid((3, 3), (1, 2))
    ax3.set_title('lag autocor. x')
    plot_lag_autocor(ax3, sample.data[:, 0])

    ax4 = plt.subplot2grid((3, 3), (2, 2))
    ax4.set_title('lag autocor. y')
    plot_lag_autocor(ax4, sample.data[:, 1])

    ax5 = plt.subplot2grid((3, 3), (1, 1), rowspan=2)
    ax5.set_title('scatter plot')
    ax5.scatter(*sample.data.transpose())
    ax5.set_aspect('equal')
    ax5.grid()
    # if target is not None:
    #     ax5 = plt.subplot2grid((3, width), (1, 1), rowspan=2)
    #     ax5.set_title("target pdf")
    #     extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    #     x = np.linspace(extent[0], extent[1], bins)
    #     y = np.linspace(extent[2], extent[3], bins)
    #     mgrid = np.meshgrid(x, y)
    #     im = ax5.imshow(target(*mgrid), extent=extent, origin='lower')
    #     plt.colorbar(im, ax=ax5)
    #     ax5.set_aspect('equal')

    fig.tight_layout()
    return fig

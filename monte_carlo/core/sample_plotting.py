import numpy as np
import matplotlib.pyplot as plt
from . import util


def plot_lag_autocor(axis, array):
    if array.size <= 300:
        ks = np.arange(array.size)
        acor = util.auto_corr(array)
    else:
        ks = np.arange(300)
        acor = util.auto_corr(array[:300])
    axis.bar(ks, acor, width=1)


def plot1d(sample, target=None):
    fig = plt.figure(figsize=(14, 7))

    # time series plot
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.set_title('time series')
    ax1.plot(sample.data)

    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax2.set_title('distribution')
    # guess good binning
    bins = int(max(10, 10 / 300 * sample.size))
    ax2.hist(sample.data, bins=bins)
    if target is not None:
        minx = min(0, np.min(sample.data))
        maxx = max(1, np.max(sample.data))
        x = np.linspace(minx, maxx, 1000)
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

    width = 2 if target is None else 3
    ax1 = plt.subplot2grid((3, width), (0, 0), colspan=width)
    ax1.set_title('time series')
    ax1.plot(sample.data[:, 0], label="x")
    ax1.plot(sample.data[:, 1], label="y")
    ax1.legend()

    # guess a good binning
    bins = int(max(10, 20 * np.sqrt(sample.size / 1000)))

    ax2 = plt.subplot2grid((3, width), (1, 0), rowspan=2)
    ax2.set_title('distribution')
    counts, xedges, yedges, im = ax2.hist2d(*sample.data.transpose(), bins=bins)
    plt.colorbar(im, ax=ax2)

    # plot
    ax3 = plt.subplot2grid((3, width), (1, width-1))
    ax3.set_title('lag autocor. x')
    plot_lag_autocor(ax3, sample.data[:, 0])

    ax4 = plt.subplot2grid((3, width), (2, width-1))
    ax4.set_title('lag autocor. y')
    plot_lag_autocor(ax4, sample.data[:, 1])

    if target is not None:
        ax5 = plt.subplot2grid((3, width), (1, 1), rowspan=2)
        ax5.set_title("target pdf")
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        x = np.linspace(extent[0], extent[1], bins)
        y = np.linspace(extent[2], extent[3], bins)
        mgrid = np.meshgrid(x, y)
        im = ax5.imshow(target(*mgrid), extent=extent)
        plt.colorbar(im, ax=ax5)

    fig.tight_layout()
    return fig

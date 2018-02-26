import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_size import AxesX, AxesY, Fraction

class PlotIndexTracker(object):
    def __init__(self, ax, mother_ax, x, y, orientation='vertical', **kwargs):
        self.ax = ax
        self.mother_ax = mother_ax
        self.x = x
        self.y = y
        self.orientation = orientation
        
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()
        
        if self.orientation == 'horizontal':
            _, self.slices = x.shape
            self.ind = self.slices//2
            self.plot, = ax.plot(x[self.ind, :], self.y, **kwargs)
            self.line, = mother_ax.plot([self.y[self.ind], self.y[self.ind]], self.ylim)
        elif self.orientation == 'vertical':
            self.slices, _ = y.shape
            self.ind = self.slices//2
            self.plot, = ax.plot(x, self.y[:, self.ind], **kwargs)
            self.line, = mother_ax.plot(self.xlim, [self.x[self.ind], self.x[self.ind]])
        
        #self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.orientation == 'horizontal':
            self.plot.set_xdata(self.x[self.ind, :])
            self.line.set_xdata([self.y[self.ind], self.y[self.ind]])
        elif self.orientation == 'vertical':
            self.plot.set_ydata(self.y[:, self.ind])
            self.line.set_ydata([self.x[self.ind], self.x[self.ind]])
            
        self.plot.axes.figure.canvas.draw()

class BarIndexTracker(object):
    def __init__(self, ax, x, y, orientation='vertical', **kwargs):
        self.ax = ax

        self.x = x
        self.y = y
        self.orientation = orientation
        
        if self.orientation == 'horizontal':
            _, self.slices = x.shape
            self.ind = self.slices//2
            self.rects = ax.barh(self.y, x[self.ind, :], **kwargs)
        elif self.orientation == 'vertical':
            self.slices, _ = y.shape
            self.ind = self.slices//2
            self.rects = ax.bar(x, self.y[:, self.ind], **kwargs)
        
        #self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.orientation == 'horizontal':
            for rect, w in zip(self.rects, self.x[self.ind, :]):
                rect.set_width(w)
        elif self.orientation == 'vertical':
            for rect, h in zip(self.rects, self.y[:, self.ind]):
                rect.set_height(h)
                
        self.ax.figure.canvas.draw()

class TrackerHandler(object):
    def __init__(self):
        self.trackerlist = []
    
    def add(self, tracker):
        self.trackerlist.append(tracker)
    
    def onscroll(self, event):
        for tracker in self.trackerlist:
            if event.inaxes == tracker.ax:
                tracker.onscroll(event)

def plot_2d(samples, target_pdf, mapping_pdf=None):
    x0_samples = [sample[0] for sample in samples]
    x1_samples = [sample[1] for sample in samples]
    
    nbins = 30
    x = y = np.arange(1./(2.*nbins),1.+1./(2.*nbins),1./nbins)
    X, Y = np.meshgrid(x, y)
    data_target = np.empty([x.size, y.size])
    #data_map = np.empty([x.size, y.size])
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            data_target[i, j] = target_pdf([x_i, y_j])
            #data_map[i, j] = (.1*np.sqrt(np.pi))**(sampler.ndim-2)*sum([alpha_i*sampler.kernels[0].mapping([x_i, y_j],channel) for channel,alpha_i in enumerate(sampler.kernels[0].alpha)])
    
    # normalize
    data_target *= len(samples)/data_target.sum()
    #data_map *= nsamples/data_map.sum()
    norm_min = 0
    norm_max = data_target.max()
    
    fig = plt.figure(0)
    outer_grid = gridspec.GridSpec(4, 4, width_ratios=[1, 3, 3, 1], height_ratios=[1, 1, 4, 4], hspace=1.0, wspace=1.0)
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, outer_grid[0:2, 0:2], hspace=0.0)
    ax_chain_x0 = plt.subplot(inner_grid[0, 0:2])
    ax_chain_x1 = plt.subplot(inner_grid[1, 0:2])
    ax_chain_2d = plt.subplot(outer_grid[0:2, 2:4], adjustable='box', aspect='equal')
    
    ax_target_2d = plt.subplot(outer_grid[2, 1:3], adjustable='box', aspect='equal')
    divider = make_axes_locatable(ax_target_2d)
    ax_target_x0 = divider.append_axes("top", size=Fraction(0.5, AxesY(ax_target_2d)), pad=0.1)
    ax_target_x1 = divider.append_axes("right", size=Fraction(0.5, AxesX(ax_target_2d)), pad=0.1)
    # make some labels invisible
    plt.setp(ax_target_x0.get_xticklabels() + ax_target_x1.get_yticklabels(),
         visible=False)
    
    ax_samples_2d = plt.subplot(outer_grid[3, 1:3], adjustable='box', aspect='equal')
    divider = make_axes_locatable(ax_samples_2d)
    ax_samples_x0 = divider.append_axes("top", size=Fraction(0.5, AxesY(ax_samples_2d)), pad=0.1, sharex=ax_samples_2d)
    ax_samples_x1 = divider.append_axes("right", size=Fraction(0.5, AxesX(ax_samples_2d)), pad=0.1, sharey=ax_samples_2d)
    # make some labels invisible
    plt.setp(ax_samples_x0.get_xticklabels() + ax_samples_x1.get_yticklabels(),
         visible=False)
    
    ax_chain_x0.set_title('Chain')
    ax_chain_x0.set_ylabel('x0')
    ax_chain_x0.set_ylim([-0.1, 1.1])
    ax_chain_x0.plot(x0_samples)
    # hide labels
    for label in ax_chain_x0.get_xticklabels():
        label.set_visible(False)
    
    ax_chain_x1.set_xlabel('t')
    ax_chain_x1.set_ylabel('x1')
    ax_chain_x1.set_ylim([-0.1, 1.1])
    ax_chain_x1.plot(x1_samples)
    
    ax_chain_2d.set_xlabel('x0')
    ax_chain_2d.set_ylabel('x1')
    plot_chain_2d = ax_chain_2d.scatter(x0_samples, x1_samples, marker='o', c=np.arange(len(samples)), cmap=plt.get_cmap('viridis'))
    ax_chain_2d.xaxis.set_ticks(np.linspace(0, 1, 5))
    ax_chain_2d.xaxis.set_ticklabels(np.linspace(0, 1, 5))
    ax_chain_2d.yaxis.set_ticks(np.linspace(0, 1, 5))
    ax_chain_2d.yaxis.set_ticklabels(np.linspace(0, 1, 5))
    ax_chain_2d.set_xlim([0, 1])
    ax_chain_2d.set_ylim([0, 1])
    cbar = fig.colorbar(plot_chain_2d, ax=ax_chain_2d)
    cbar.set_label('sample number', rotation=270, labelpad=10)
    
    ax_target_x0.set_title('Target PDF')
    ax_target_x0.set_ylim([0, data_target.max()])
    tracker_target_x0 = PlotIndexTracker(ax_target_x0, ax_target_2d, x, data_target)
    trackerhandler = TrackerHandler()
    trackerhandler.add(tracker_target_x0)
    
    ax_target_x1.set_xlim([0, data_target.max()])
    tracker_target_x1 = PlotIndexTracker(ax_target_x1, ax_target_2d, data_target, y, orientation='horizontal')
    trackerhandler.add(tracker_target_x1)
    
    ax_target_2d.set_xlabel('x0')
    ax_target_2d.set_ylabel('x1')
    plot_target_2d = ax_target_2d.imshow(data_target.transpose(), origin='lower', extent=[0, 1, 0, 1], vmin=norm_min, vmax=norm_max, cmap=plt.get_cmap('viridis'), interpolation='none')
    ax_target_2d.xaxis.set_ticks(np.linspace(0, 1, 5))
    ax_target_2d.xaxis.set_ticklabels(np.linspace(0, 1, 5))
    ax_target_2d.yaxis.set_ticks(np.linspace(0, 1, 5))
    ax_target_2d.yaxis.set_ticklabels(np.linspace(0, 1, 5))
    cbar = fig.colorbar(plot_target_2d, ax=ax_target_2d)
    cbar.set_label('Frequency', rotation=270, labelpad=10)
    
    ax_samples_2d.set_xlabel('x0')
    ax_samples_2d.set_ylabel('x1')
    hist, _, _, plot_samples_2d = ax_samples_2d.hist2d(x0_samples, x1_samples, bins=nbins, range=[[0, 1], [0, 1]], vmin=norm_min, vmax=norm_max, cmap=plt.get_cmap('viridis'))
    ax_samples_2d.xaxis.set_ticks(np.linspace(0, 1, 5))
    ax_samples_2d.xaxis.set_ticklabels(np.linspace(0, 1, 5))
    ax_samples_2d.yaxis.set_ticks(np.linspace(0, 1, 5))
    ax_samples_2d.yaxis.set_ticklabels(np.linspace(0, 1, 5))
    cbar = fig.colorbar(plot_samples_2d, ax=ax_samples_2d)
    cbar.set_label('Frequency', rotation=270, labelpad=10)
    
    ax_samples_x0.set_title('Samples')
    ax_samples_x0.set_ylim([0, hist.max()])
    tracker_samples_x0_bar = BarIndexTracker(ax_samples_x0, x, hist, width=1/nbins)
    tracker_samples_x0_plot = PlotIndexTracker(ax_samples_x0, ax_samples_2d, x, data_target, color='orange')
    trackerhandler.add(tracker_samples_x0_bar)
    trackerhandler.add(tracker_samples_x0_plot)
    
    ax_samples_x1.set_xlim([0, hist.max()])
    tracker_samples_x1_bar = BarIndexTracker(ax_samples_x1, hist, y, orientation='horizontal', height=1/nbins)
    tracker_samples_x1_plot = PlotIndexTracker(ax_samples_x1, ax_samples_2d, data_target, y, orientation='horizontal', color='orange')
    trackerhandler.add(tracker_samples_x1_bar)
    trackerhandler.add(tracker_samples_x1_plot)
    
    fig.canvas.mpl_connect('scroll_event', trackerhandler.onscroll)
    
    plt.show()

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
            self.slices,_ = x.shape
            self.ind = self.slices//2
            self.plot, = ax.plot(x[self.ind, :], self.y, **kwargs)
            binwith = 1/self.slices
            self.xt = np.linspace(binwith/2,1-binwith/2,self.slices)
            self.line, = mother_ax.plot([self.xt[self.ind], self.xt[self.ind]], self.ylim)
        elif self.orientation == 'vertical':
            _,self.slices = y.shape
            self.ind = self.slices//2
            self.plot, = ax.plot(x, self.y[:, self.ind], **kwargs)
            binwith = 1/self.slices
            self.yt = np.linspace(binwith/2,1-binwith/2,self.slices)
            self.line, = mother_ax.plot(self.xlim, [self.yt[self.ind], self.yt[self.ind]])
        
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
            self.line.set_xdata([self.xt[self.ind], self.xt[self.ind]])
        elif self.orientation == 'vertical':
            self.plot.set_ydata(self.y[:, self.ind])
            self.line.set_ydata([self.yt[self.ind], self.yt[self.ind]])
            
        self.plot.axes.figure.canvas.draw()

class BarIndexTracker(object):
    def __init__(self, ax, x, y, orientation='vertical', **kwargs):
        self.ax = ax

        self.x = x
        self.y = y
        self.orientation = orientation
        
        if self.orientation == 'horizontal':
            self.slices,_ = x.shape
            self.ind = self.slices//2
            self.rects = ax.barh(self.y, x[self.ind, :], **kwargs)
        elif self.orientation == 'vertical':
            _,self.slices = y.shape
            print(self.slices)
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

def plot_2d(hist, data_target, title="",labelx="x",labely="y",labelz="Z",xticks=None,yticks=None):
   
    shape=np.shape(hist)
    if isinstance(data_target,float):
        data_target = np.full(shape,data_target)

    nbinsX = shape[0]
    nbinsY = shape[1]
    x = np.arange(1./(2.*nbinsX),1.+1./(2.*nbinsX),1./nbinsX)

    y = np.arange(1./(2.*nbinsY),1.+1./(2.*nbinsY),1./nbinsY)

    # normalize 
    norm_min = 0
    norm_max = hist.max()
    
    fig = plt.figure(0)
    outer_grid = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1], hspace=1.0, wspace=1.0)    

        
    ax_samples_2d = plt.subplot(outer_grid[0,0], adjustable='box', aspect='equal')
    divider = make_axes_locatable(ax_samples_2d)
    ax_samples_x0 = divider.append_axes("top", size=Fraction(0.5, AxesY(ax_samples_2d)), pad=0.1, sharex=ax_samples_2d)
    ax_samples_x1 = divider.append_axes("right", size=Fraction(0.5, AxesX(ax_samples_2d)), pad=0.1, sharey=ax_samples_2d)
    # make some labels invisible
    plt.setp(ax_samples_x0.get_xticklabels() + ax_samples_x1.get_yticklabels(),
         visible=False)
     
    trackerhandler = TrackerHandler()
            
    ax_samples_2d.set_xlabel(labelx)
    ax_samples_2d.set_ylabel(labely)
    _, _, _, _ = ax_samples_2d.hist2d([0,1], [0,1], bins=shape, range=[[0,1],[0,1]], vmin=norm_min, vmax=norm_max, cmap=plt.get_cmap('viridis'))

    plot_samples_2d = ax_samples_2d.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], vmin=norm_min, vmax=norm_max, cmap=plt.get_cmap('viridis'), interpolation='none')

    ax_samples_2d.xaxis.set_ticks(np.linspace(0, 1, 5))
    ax_samples_2d.xaxis.set_ticklabels(np.linspace(0, 1, 5))
    ax_samples_2d.yaxis.set_ticks(np.linspace(0, 1, 5))
    ax_samples_2d.yaxis.set_ticklabels(np.linspace(0, 1, 5))

    if xticks is not None:
        ax_samples_2d.xaxis.set_ticks(x[::nbinsX//4])
        ax_samples_2d.xaxis.set_ticklabels(xticks[::nbinsX//4])
    if yticks is not None:
        ax_samples_2d.yaxis.set_ticks(y[::nbinsY//4])
        ax_samples_2d.yaxis.set_ticklabels(yticks[::nbinsY//4])
    cbar = fig.colorbar(plot_samples_2d, ax=ax_samples_2d)
    cbar.set_label(labelz, rotation=270, labelpad=10)
    
    ax_samples_x0.set_title(title)
    ax_samples_x0.set_ylim([0, hist.max()])
    tracker_samples_x0_bar = BarIndexTracker(ax_samples_x0, x, hist, width=1/nbinsX)
    tracker_samples_x0_plot = PlotIndexTracker(ax_samples_x0, ax_samples_2d, x, data_target, color='orange')
    trackerhandler.add(tracker_samples_x0_bar)
    trackerhandler.add(tracker_samples_x0_plot)
    
    ax_samples_x1.set_xlim([0, hist.max()])
    tracker_samples_x1_bar = BarIndexTracker(ax_samples_x1, hist, y, orientation='horizontal', height=1/nbinsY)
    tracker_samples_x1_plot = PlotIndexTracker(ax_samples_x1, ax_samples_2d, data_target, y, orientation='horizontal', color='orange')
    trackerhandler.add(tracker_samples_x1_bar)
    trackerhandler.add(tracker_samples_x1_plot)
    
    fig.canvas.mpl_connect('scroll_event', trackerhandler.onscroll)
    
    plt.show()

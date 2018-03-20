import numpy as np
import matplotlib.pyplot as plt

from plot.plot2d import plot_2d

def extract(data,varZ, varX, varY,):
    varX_range=np.sort(np.unique(data[varX]))
    varY_range=np.sort(np.unique(data[varY]))
    a = np.empty([len(varX_range),len(varY_range)],dtype=np.ndarray)
    
    for iX,vX in enumerate(varX_range):
        for iY,vY in enumerate(varY_range):
            a[iX,iY] = data[np.logical_and(data[varX]==vX,data[varY]==vY)][varZ][0]

    return np.array(a.tolist()),varX_range,varY_range
    
def plot_parm(filename,varZ,varX,varY):
    data = np.load(filename)
    matrix,xticks,yticks = extract(data,varZ,varX,varY)
    
    if varZ == "chiq":
        plot_2d(matrix[:,:,1],0.05,r'$\chi^2$',varX,varY,'p-value',xticks=xticks,yticks=yticks)
    if varZ == "chiqp":
        plot_2d(matrix[:,:,0,1],0.05,r'$\chi^2$ on axis 0',varX,varY,'p-value',xticks=xticks,yticks=yticks)
        plot_2d(matrix[:,:,1,1],0.05,r'$\chi^2$ on axis 1',varX,varY,'p-value',xticks=xticks,yticks=yticks)
    if varZ == "ks":
        plot_2d(matrix[:,:,0,0],matrix[:,:,0,1],"ks on axis 0",varX,varY,r'$d_{max}$',xticks=xticks,yticks=yticks)
        plot_2d(matrix[:,:,1,0],matrix[:,:,1,1],"ks on axis 1",varX,varY,r'$d_{max}$',xticks=xticks,yticks=yticks)    


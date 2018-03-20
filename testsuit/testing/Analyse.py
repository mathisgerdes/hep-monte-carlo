import numpy as np
import os



from statistics.print_statistics import print_statistics
from statistics.effective_sample_size import effective_sample_size

from testing.stats import chiq, chiqp, ks


def getListOfSamples(startDirectory):
    Sampels =[]
    listOfContent = os.listdir(startDirectory)
    for item in listOfContent:
        if os.path.isdir(item):
            Sampels.append(startDirectory+"/"+item)
        elif item.endswith(".npy"):
            Sampels.append(startDirectory+"/"+item)
    return Sampels


def loopSamples(Directory,reference,bins=10,maxlag=25):
    listOfSamples = getListOfSamples(Directory)
    Results = []
    for item in listOfSamples:
        if isinstance(item,list):
            Results.append(loopSamples(item))
        else:
            Results.append(doAnalysis(item,reference,bins,maxlag))
    return np.concatenate(Results)
    

def doAnalysis(pathToSample,pathToReference,bins,maxlag):
    
    # load reference
    truth = np.load(pathToReference)
    truth,edges = np.histogramdd(truth[0],bins=bins)

    # extract parameters from filename
    temp = pathToSample[:-4].split("-")
    values = (pathToSample,)
    dtypes = [("Sample",'|O')]
    for parameter in temp[1:]:
        name,value = parameter.split("_")
        try:
            value=int(value)
            values+=(value,)
            dtypes.append((name,int))
        except ValueError:
            value=float(value)
            values+=(value,)
            dtypes.append((name,float))

    sample = np.load(pathToSample)[()]

    mean          = sample["mean"]
    variance      = sample["variance"]
    n_taget_calls = sample["n_target_calls"]
    runtime       = sample["runtime"]
    samples       = sample["samples"]

# in recarray speichern
    samplesize = int(round(len(samples)/maxlag))
    
    values = (*values,samplesize)
    dtypes.append(("size",int))


    Results = []
    first = True
    for lag in range(1,maxlag+1):
        values_lag = values
        dtypes_lag = dtypes[:]

        discrete_sample, _ = np.histogramdd((samples[::lag])[:samplesize],bins=edges)
 
        values_lag = (*values_lag,lag)
        dtypes_lag.append(("lag",int))


        values_lag = (*values_lag,chiq(discrete_sample,truth))
        dtypes_lag.append(("chiq",list))
    
        values_lag = (*values_lag,chiqp(discrete_sample,truth))
        dtypes_lag.append(("chiqp",list))

        
        values_lag = (*values_lag,ks(discrete_sample,truth))
        dtypes_lag.append(("ks",list))

        temp = np.array(values_lag,dtype=dtypes_lag)



        if first:
            Results = temp
            first = False
        else:
            Results = np.append(Results,temp)

    
    return Results


################################################################################


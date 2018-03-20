import numpy as np
from testing.Analyse import loopSamples
from plot.plot_parm import plot_parm


# config
ReferencePath = "reference_camel-1M.2d.npy"
SamplePath    = "cv"
bins          = 15
maxlag        = 20

Output_FileName = "Camel2d-cv"

Results = loopSamples(SamplePath,ReferencePath,bins,maxlag)

np.save(Output_FileName,Results)

         #filename               test   var1  var2
plot_parm(Output_FileName+".npy","chiq","beta","lag")
plot_parm(Output_FileName+".npy","chiqp","beta","lag")
plot_parm(Output_FileName+".npy","ks"  ,"beta","lag")

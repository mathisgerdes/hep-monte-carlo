import inspect
from samplers.mcmc.mixed import MixedSampler


import numpy as np

def MixedSamplerSetUp(samplers=[],args=[],start=0.5,target_pdf=None,ndim=None,nburnin = None, beta=None,**kwargs):

    kernels = []
    for sampler,kwargs in zip(samplers,args):
        print(sampler.__name__)
        kernels.append(sampler(**kwargs))

        if ndim is not None:
            kwargs.update({"ndim":ndim})
        if target_pdf is not None:
            kwargs.update({"target_pdf":target_pdf})

        if nburnin is not None:
            print("StarteBurnIn")
            kernels[-1].sample(nburnin,start)

        # disable adaptive sampler for the moment
        try:
            kernels[-1].is_adaptive=False
        except AttributeError:
            print("Cannot set is_adaptiv=False")

        
    if beta is None:
        sampler_weights = [0.5,0.5]
        print("No beta specified use default 0.5")
    else:
         sampler_weights = [beta,1-beta]    

    return MixedSampler(kernels,sampler_weights)
            




import random

def getfrac(dset, exs=1000, log=True): 
    inds = list(range(len(dset)))
    sampinds = random.sample(inds, exs)
    if log:
        print(sampinds[:20])
    return dset.select(sampinds)

import random

def getfrac(dset, exs=1000, log=True): 
    inds = list(range(len(dset)))
    sampinds = random.sample(inds, exs)
    if log:
        print(sampinds[:20])
    return dset.select(sampinds)

def rev_lab(ex):
    tj = ex['response_j']
    tjs = ex['score_j']
    ex['response_j'] = ex['response_k']
    ex['score_j'] = ex['score_k'] *-1
    ex['response_k'] = tj
    ex['score_k'] = tjs *-1
    ex['magnitude'] = ex['score_j'] - ex['score_k']
    return ex
    
def reverse_labels(dset):
    return dset.map(rev_lab, num_proc=10)
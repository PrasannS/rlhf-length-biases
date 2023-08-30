from transformers import AutoTokenizer
import pandas as pd
from rlhfutils.eval_utils import getapfsft, tok_dist
import matplotlib.pyplot as plt
from rlhfutils.debug_utils import load_rm, progress_rm
import argparse


# replace all wgptouts with corresponding stack QA format (RM input format)
def setall(l):
    newl = []
    try:
        for ind in l:
            newl.append(getapfsft(ind, True))
            #print(0)
    except:
        return None
    return newl

def splitall(l):
    try: 
        return [s.split("Answer:")[1] for s in l]
    except:
        return None

def getfulldist(lcol):
    hist = []
    for l in lcol:
        hist.extend(l)
    return hist

def compdist(lcol, slen):
    res = []
    tmp = []
    for i in range(len(lcol)):
        tmp.append(lcol[i])
        if len(tmp)%slen==0:
            res.append(tmp)
            tmp = []
    return res
    
def procall(indf, toker, needset=True):
    if needset:
        indf['response'] = [setall(s) for s in indf['response']]
    indf = indf.dropna()
    indf['answers'] = [splitall(s) for s in indf['response']]
    indf = indf.dropna()
    indf['atoks'] = [tok_dist(s, toker) for s in list(indf['answers'])]
    indf['ttoks'] = [tok_dist(s, toker) for s in list(indf['response'])]
    return indf

def main(args):
    "../trl-general/genouts/generated_stackmultisampset.jsonl"
    "../trl-general/genouts/generated_wgptmultisampset.jsonl"
    stack = "stack" in args.rmname
    outdf = pd.read_json(args.inpf, orient='records', lines=True)
    if stack:
        tok = AutoTokenizer.from_pretrained("../stack-llama/models/sft/")
        outdf = procall(outdf, tok, False)
    else:
        tok = AutoTokenizer.from_pretrained("../webgpt-llama/models/sft10k/")
        outdf = procall(outdf, tok, True)
    tok, rm, kwargs = load_rm(args.rmname, args.device)
    allresps = getfulldist(outdf.response)
    allscos = progress_rm(allresps, rm, kwargs)
    scos = compdist([a[0]['score'] for a in allscos], 8)
    outdf[args.rmname.split("/")[-1]] = scos
    outdf.to_json(args.rmname.split("/")[-1]+".jsonl", lines=True, orient='records')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='My Python script.')
    # TODO fix up arg names
    parser.add_argument('rmname', type=str, help='base model checkpoint is trained on')
    parser.add_argument('inpf', type=str, help='bottom of range to generate for')
    parser.add_argument('device', type=int, help='outputs per prompt')
    
    progargs = parser.parse_args()
    
    main(progargs)
    
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from rlhfutils.eval_utils import getapfsft, tok_dist
import matplotlib.pyplot as plt
from rlhfutils.debug_utils import load_rm, progress_rm
import argparse
import nltk
from nltk.tokenize import sent_tokenize
import random
import torch
from dpo_eval import calculate_log_likelihoods
nltk.download('punkt')

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

def shuffle_sents(paragraph):
    # Use NLTK to split the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    # Shuffle the sentences
    random.shuffle(sentences)
    
    # Combine the shuffled sentences back into a paragraph
    shuffled_paragraph = ' '.join(sentences)
    
    return shuffled_paragraph

def shuffle_row_resp(row):
    # shuffle answers for 
    shuffans = [shuffle_sents(r) for r in row['answers']]
    return [row['response'][i][:-len(row['answers'][i])]+shuffans[i] for i in range(len(shuffans))]


def main(args):
    "../trl-general/genouts/generated_stackmultisampset.jsonl"
    "../trl-general/genouts/generated_wgptmultisampset.jsonl"
    stack = "stack" in args.rmname
    outdf = pd.read_json(args.inpf, orient='records', lines=True)    
    if args.lim > 0:
        outdf = outdf.iloc[:args.lim]
    tokdir = args.rmname if "nobase" in args.basemodel else args.basemodel
    if stack:
        tok = AutoTokenizer.from_pretrained(tokdir)
        outdf = procall(outdf, tok, False)
    else:
        tok = AutoTokenizer.from_pretrained(tokdir)
        outdf = procall(outdf, tok, True)
        
    allresps = getfulldist(outdf.response)
    # if we want, we can score perturbed data (shuffle sentences via nltk)
    # TODO do a double check on whether removing truncated sentence from output helps (APEval)
    if args.shuffle>0:
        outdf['response'] = [shuffle_row_resp(r) for _, r in outdf.iterrows()]
    
    print(allresps[0])
    if 'dpo' in args.rmname or 'tulu' in args.rmname:
        tokenizer = AutoTokenizer.from_pretrained(args.rmname)
        model = AutoModelForCausalLM.from_pretrained(args.rmname, device_map="auto")
        model.eval()
        allresps = [r.replace("Question: ", "<|user|>\n").replace("\n\nAnswer: ", "\n<|assistant|>\n") for r in allresps]
        
        scos = calculate_log_likelihoods(allresps, tokenizer, model)
        scos = compdist(scos, 4)
    else:
        tok, rm, kwargs = load_rm(args.rmname, args.device, True, args.basemodel)
        if kwargs:
            with torch.no_grad():
                allscos = progress_rm(allresps, rm, kwargs)
        # TODO compdist set to 4
        scos = compdist([a[0]['score'] for a in allscos], 4)
    outdf['scores'] = scos
    if args.outputdir=='default':
        
        outdf.to_json(args.rmname.split("/")[-1]+".jsonl", lines=True, orient='records')
    else:
        outdf.to_json(args.outputdir, lines=True, orient='records')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='My Python script.')
    # TODO fix up arg names
    parser.add_argument('--rmname', type=str, help='base model checkpoint is trained on')
    parser.add_argument('--inpf', type=str, help='bottom of range to generate for')
    parser.add_argument('--device', type=int, help='outputs per prompt')
    parser.add_argument('--lim', type=int, help="whatever")
    parser.add_argument('--shuffle', type=int, help='whether to shuffle sentences before scoring or not')
    parser.add_argument('--basemodel', type=str, help='base model for RM', default="nobase")
    parser.add_argument('--outputdir', type=str, help='base model for RM', default="default")
    
    progargs = parser.parse_args()
    
    main(progargs)
    
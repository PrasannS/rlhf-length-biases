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
from tqdm import tqdm
nltk.download('punkt')

# replace all wgptouts with corresponding stack QA format (RM input format)
def setall(l, tok, args):
    newl = []
    try:
        for ind in l:
            newl.append(getapfsft(ind, True, tok, args.maxlen))
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
    
def procall(indf, toker, args, needset=True):
    if needset:
        indf['response'] = [setall(s, toker, args) for s in indf['response']]
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
    for c in args.cklist:
        inps = [args.inpf]
        if "," in args.inpf:
            inps = args.inpf.split(",")
        for inpfile in inps:
            stack = "stack" in args.rmname
            outdf = pd.read_json(inpfile, orient='records', lines=True)    
            if args.lim > 0:
                outdf = outdf.iloc[:args.lim]
            # make it checkpoint specific
            tokdir = args.rmname+c if "nobase" in args.basemodel else args.basemodel
            if stack:
                tok = AutoTokenizer.from_pretrained(tokdir)
                outdf = procall(outdf, tok, args, False)
            else:
                tok = AutoTokenizer.from_pretrained(tokdir)
                outdf = procall(outdf, tok, args, True)
                
            allresps = getfulldist(outdf.response)
            # if we want, we can score perturbed data (shuffle sentences via nltk)
            # TODO do a double check on whether removing truncated sentence from output helps (APEval)
            if args.shuffle>0:
                outdf['response'] = [shuffle_row_resp(r) for _, r in outdf.iterrows()]
            
            print(allresps[0])
            if 'dpo' in args.rmname or 'tulu' in args.rmname:
                tokenizer = AutoTokenizer.from_pretrained(args.rmname+c)
                model = AutoModelForCausalLM.from_pretrained(args.rmname+c, device_map="auto")
                model.eval()
                allresps = [r.replace("Question: ", "<|user|>\n").replace("\n\nAnswer: ", "\n<|assistant|>\n") for r in allresps]
                
                scos = calculate_log_likelihoods(allresps, tokenizer, model)
                scos = compdist(scos, len(outdf.response.iloc[0]))
            else:
                tok, rm, kwargs = load_rm(args.rmname+c, args.device, True, args.basemodel, True, args.tokenwise)
                
                if args.tokenwise: 
                    scos = []
                    with torch.no_grad():
                        BSIZE= 8  # TODO make this a hyperparam
                        for r in tqdm(range(0, len(allresps), BSIZE)):
                            inputs = tok(allresps[r:r+BSIZE], padding=True, truncation=True, return_tensors="pt").to(rm.device)
                            outputs_j = rm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                            probs_j = torch.sigmoid(outputs_j.logits)
                            rewards_j = probs_j.sum(dim=1)
                            scos.extend([s[0][0] for s in rewards_j.unsqueeze(-1).detach().tolist()])
                        assert len(scos)==len(allresps)
                    scos = compdist(scos, len(outdf.response.iloc[0]))
                else:
                    if kwargs:
                        with torch.no_grad():
                            allscos = progress_rm(allresps, rm, kwargs)
                    # TODO compdist set to 4
                    scos = compdist([a[0]['score'] for a in allscos], len(outdf.response.iloc[0]))
            outdf['scores'] = scos
            if args.outputdir=='default':
                
                outdf.to_json(args.rmname.split("/")[-1]+".jsonl", lines=True, orient='records')
            else:
                outdf.to_json(args.outputdir+c, lines=True, orient='records')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='My Python script.')
    # TODO fix up arg names
    parser.add_argument('--rmname', type=str, help='base model checkpoint is trained on')
    parser.add_argument('--inpf', type=str, help='input file')
    parser.add_argument('--device', type=int, help='outputs per prompt')
    parser.add_argument('--lim', type=int, help="whatever")
    parser.add_argument('--shuffle', type=int, help='whether to shuffle sentences before scoring or not')
    parser.add_argument('--basemodel', type=str, help='base model for RM', default="nobase")
    parser.add_argument('--outputdir', type=str, help='base model for RM', default="default")
    parser.add_argument('--maxlen', type=int, help='max len to use for outputs', default=-1)
    parser.add_argument("--cklist", type=str, default=[""], help='list of checkpoint numbers we want to do stuff for')
    parser.add_argument("--tokenwise", type=bool, default=False, help="whether to use tokenwise formulation or not")

    
    progargs = parser.parse_args()
    
    main(progargs)
    
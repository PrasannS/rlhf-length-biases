# Run code to get adversarial test numbers for the 3 different RMs at different points. Get the data at the end. 
import os

import torch

from tqdm import tqdm
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM



from transformers import AutoTokenizer

from rm_grad_inputs import propose_new_sequence

import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/rewardsanity")
tokenizer.pad_token = tokenizer.eos_token

def count_words(indf):
    wcnt = [len(s.split(" ")) for s in indf['question']]
    return wcnt

def load_alldfs(base="/home/prasann/Projects/tfr-decoding/trlx_train/use_outs/"):
    alldfs = {}
    for f in os.listdir(base):
        if f[0]=='.':
            continue
        tmp = pd.read_json(base+f, lines=True, orient='records')
        #if "Question:" in tmp['response'][0]:
        #    tmp['response'] = [r[len(q):] for r, q in zip(tmp['response'], tmp['question'])]
        #if "Question:" in tmp['question'][0]:
        #    tmp['question'] = [s[len("Question: "):-1*len("\n\nAnswer: ")] for s in tmp['question']]
        # constrain to only 200 examples for everything
        tmp = tmp.loc[:70]
        tmp['wcnts'] = count_words(tmp)
        tmp = tmp[tmp['wcnts']<500].reset_index()
        alldfs[f.replace("generated_", "").replace(".jsonl", "")] = tmp
    return alldfs

def overkeys(adfs, fpaths):
    usekeys = []
    for f in fpaths:
        for k in adfs.keys():
            if f in k:
                usekeys.append(k)
    return usekeys

def loadrm(rname, device):
    return AutoModelForSequenceClassification.from_pretrained(
        rname, num_labels=1, device_map={"": 0},
        load_in_8bit=True,
    )
    
def run_rmscos(rmname, sftmod, fchecks, adfs):
    rmodel = loadrm(rmname)
    # we're going to run adversarial stuff only on necessary code
    usekeys = overkeys(adfs, fchecks)
    
    # load in dataframes
    for uk in tqdm(usekeys):
        indf = adfs[uk]
        tmpres = []
        tmpf = "attackouts/"+fchecks[0]+"_"+uk+".jsonl"
        # Check if the file already exists
        if os.path.exists(tmpf):
            # Load the saved rows
            existing_data = pd.read_json(tmpf, orient='records', lines=True)
            num_existing_rows = len(existing_data)
            tmpres = existing_data.to_dict('records')
            print("loading in from existing file")
        else:
            num_existing_rows = 0
            tmpres = []

        # Start processing from where we left off
        for ind, row in tqdm(indf.iterrows()):
            if ind < num_existing_rows:
                print("skip")
                continue  # Skip rows that have already been processed

            try:
                rmodel.zero_grad()
                torch.cuda.empty_cache()
                # go through sequence proposition process
                try:
                    result = propose_new_sequence(row['response'], 5, 10, tokenizer, rmodel, sftmod, 1)
                    tmpres.append(result)
                except:
                    tmpres.append({
                    'origseq':None,
                    'origsco':None,
                    'bestseqs':None,
                    'bestscos':None
                })
                # save stuff at each step
                pd.DataFrame(tmpres).to_json(tmpf, orient='records', lines=True)
            except RuntimeError as e:
                # in case we run out of space somehow
                del rmodel
                torch.cuda.empty_cache()
            

if __name__=="__main__":
    
    # rmnames = ["../stack-llama/models/rmodel"]
    rmnames = ["/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/rewardsanity/"]
    
    fcheck = [
        ["sft"],
        ["mix", "chosen", "reject", "sft"],
        ["da", "chosen", "reject", "sft"]
    ]
    
    # add in sft model for extra weight
    sftmodel = AutoModelForCausalLM.from_pretrained(
        "../stack-llama/models/sft/",
        load_in_8bit=True, # re-enable for llama model
        device_map={"": 1},
    )
    sftmodel.eval()
    alldfs = load_alldfs()
    print(alldfs.keys())
    
    for i in range(len(rmnames)):
        run_rmscos(rmnames[i], sftmodel, fcheck[i], alldfs)
    
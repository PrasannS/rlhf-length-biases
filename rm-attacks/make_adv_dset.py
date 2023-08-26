from datasets import load_dataset
from tqdm import tqdm
from transformers import  AutoModelForSequenceClassification
from rm_grad_inputs import propose_new_sequence
import pandas as pd
from debug_utils import load_rm as load_rm_pipe
import random
import torch
import os
import argparse


def combresp(example, juse=False):
    ex = {}
    # randomly use either a chosen or rejected string, ideally this should give some more variance / cover some gaps
    # NOTE what if it's necessary to generate from SFT model for stuff to work?
    jbool = bool(random.getrandbits(1))
    if juse:
        jbool = juse
    if jbool:
        ex['instr'] =  "Question: " + example['question'] + "\n\nAnswer: " + example['response_j']
    else:
        ex['instr'] =  "Question: " + example['question'] + "\n\nAnswer: " + example['response_k']
    return ex

def load_reward(rname, device):
    return AutoModelForSequenceClassification.from_pretrained(
        rname, num_labels=1, device_map={"": device},
        load_in_8bit=True,
    )
    
def make_rmset(rmpath, indset, N, B, sdevice, savepath):
    # get the pipeline for fast score inference, and the model for gradients
    print("loading models")
    tok, rpipe, kwargs = load_rm_pipe(rmpath, sdevice+1)
    rmodel = load_reward(rmpath, sdevice)
    allres = [] 
    if os.path.exists(savepath):
        # Load the saved rows
        existing_data = pd.read_json(savepath, orient='records', lines=True)
        num_existing_rows = len(existing_data)
        allres = existing_data.to_dict('records')
        print("loading in from existing file")
    else:
        num_existing_rows = 0
        allres = []
    ind = 0
    for row in tqdm(indset):
        # skip over row that cause error
        if ind < num_existing_rows+random.randint(1,10):
            print("skip")
            ind = ind+1
            continue 
        try:
            rmodel.zero_grad()
            torch.cuda.empty_cache()
            restmp = propose_new_sequence(row['instr'], N , B, tok, rmodel, (rpipe, kwargs), sftmodel=None, alpha=0, verbose=False)
            print(restmp['origsco'], " ", restmp['bestscos'])
            allres.append(restmp)
            pd.DataFrame(allres).to_json(savepath, orient="records", lines=True)
        except:
            # in case we run out of space somehow
            del rmodel
            torch.cuda.empty_cache()
            rmodel = load_reward(rmpath, sdevice)
        ind = ind+1
    return allres
    
def main(args):
    print("process ", str(args.process))
    proc = args.process
    MAXROUNDS = 5
    ROUNDCANDS = 10
    SDEVICE=0
    DSTART= [0,20000]
    fname = ["attackouts/dset/dsetshuff.jsonl", "attackouts/dset/dsetshuff2.jsonl"]
    
    rmpath = "../stack-llama/models/rewardsanity"
    # load reward model (sanity check RM)
    print("loading dataset")
    # get starter dataset (new data) to mess around with while using the RM
    startdset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
    startdset = startdset.shuffle(seed=0)
    startdset = startdset.select(range(DSTART[proc],DSTART[proc]+20000))
    
    print("mapping")
    # take random strings from either chosen or rejected
    startdset = startdset.map(combresp)
    startdset = startdset.filter(lambda x: len(x["instr"]) < 4000, batched=False)
    
    # generate adversarial dataset, store automatically
    make_rmset(rmpath, startdset, MAXROUNDS, ROUNDCANDS, SDEVICE, fname[proc])
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='My Python script.')
    parser.add_argument('process', type=int, help='an integer argument')

    progargs = parser.parse_args()
    main(progargs)
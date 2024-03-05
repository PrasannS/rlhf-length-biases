from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import random


# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import set_seed
from contextlib import nullcontext
from torch.utils.data import DataLoader
from datasets import Dataset
import json


from rlhfutils.rl_utils import (
    load_models, 
)
from rlhfutils.api_utils import ScriptArguments

from rlhfutils.data import load_manual, tokenize_dset
import torch
from torch import nn

import threading

from rlhfutils.rmcode import RewardDataCollatorWithPadding
import rlhfutils.rewards as rw
from rlhfutils.rewards import get_synth_rewards

lock = threading.Lock()

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

# TODO give this thing its own params

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)
random.seed(script_args.seed)
print(script_args)
    
def add_row_index(example, idx):
    example['row_index'] = idx
    return example

if script_args.trainable:
    # load in the model (trainable, no FA)
    tokenizer, reward_model = load_models(script_args, "train")
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=script_args.learning_rate)
    # get the data so that we can update things continually (NOTE may not use)
    train_dataset, evald = load_manual(script_args.dataset_name, "")
    train_dataset = train_dataset.shuffle(seed=100)
    train_dataset = train_dataset.map(add_row_index, with_indices=True)
    evald = evald.map(add_row_index, with_indices=True)
    train_dataset, _ = tokenize_dset(train_dataset, evald, script_args, tokenizer)
    print("updated length is ", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=script_args.batch_size, shuffle=True, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
    print("total batches is", len(train_dataloader))
    loaddata = iter(train_dataloader)
    if "contrastivedistill" in script_args.goldreward:
        print("loading the RM")
        rw.likemod = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        rw.liketok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        rw.slikemod = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        rw.sliketok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if script_args.noupdates:
        print("setting rm to not be updatable")
        reward_model.eval()
else:
    # NOTE handle loading everything in, since hyperparams are same for every setting more or less
    tokenizer, reward_model = load_models(script_args, "rm")

tokenizer.pad_token = tokenizer.eos_token
    
# indiv = script_args.indiv

# TODO assume that RM strings are passed in with the right format

app = Flask(__name__)

call_count = 0
label_count = 0
all_texts = []
callratio = 3 # TODO may want to set code up to make this another param for process flexibility
device = reward_model.device
extradata = []
threshsum = 20 # manually setting things here
lmb = 0.9
labelthresh = 0.3
goldlabels = 0
heursteps=50
redo_batches = 5

logdata = []
reuses = {}

print("size of tokd thing, ", tokenizer("hi there", padding=True, truncation=True, return_tensors='pt').input_ids.shape)
    
@app.route('/train', methods=['POST'])
def train():
    global call_count, all_texts, loaddata, threshsum, goldlabels, extradata, logdata, reuses, label_count
    
    
    # for thread safety
    with lock:
        call_count += 1
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])
        # for logging / later use
        all_texts.extend(input_texts)
        scores = []
        
        print("received")
        # TODO depending on speed may need to turn off
        for i in range(0, len(input_texts), script_args.batch_size):
            varupdata = (call_count%callratio)==0
            # if we want to do the variance updates
            ncont = False # ((varupdata==False) and (script_args.noupdates==False))
            with torch.no_grad():
                # pad new data / get scores for it (assume that it's all paired)
                inps = tokenizer(input_texts[i:i+script_args.batch_size], padding=True, truncation=True, return_tensors="pt").to(reward_model.device)
                rewards = reward_model(input_ids=inps.input_ids, attention_mask=inps.attention_mask)[0]
                scores.extend(rewards.detach().squeeze(-1).tolist())
            
            if varupdata:
                tottmp = 0
                acc = 0
                for ind in range(0, len(rewards), 2):
                    newgs = get_synth_rewards(input_texts[i+ind:i+ind+2], script_args.goldreward)
                    tmp = {
                        'texts':input_texts[i+ind:i+ind+2],
                        'rewards':[float(f) for f in rewards[ind:ind+2]],
                        'golds':newgs,
                        'thresh':0,
                        'step':call_count
                    }
                    if newgs[0]!=newgs[1]:
                        tottmp+=1
                        acc+= 1 if (rewards[ind]>rewards[ind+1])==(newgs[0]>newgs[1]) else 0
                    # TODO this needs to have an abs on it I think? Currrently most examples are getting used
                    if (rewards[ind]-rewards[ind+1]) < labelthresh*threshsum:
                        label_count+=1
                        if label_count>=script_args.stopupdates:
                            # NOTE if we label past a certain limit then stop grad updates
                            script_args.noupdates=True
                            reward_model.eval()
                        tmp['thresh'] = float(labelthresh*threshsum)
                        # TODO this may have been a big bug? 
                        
                        print("new rewards ", newgs)
                        # track how much extra data is needed (TODO at the beginning this will make some noise)
                        goldlabels +=1
                        if newgs[0]>newgs[1]:
                            jind = ind
                            kind = ind+1
                        elif newgs[1]>newgs[0]:
                            jind = ind+1
                            kind = ind
                        else: 
                            # TODO need to sanity check this logging setup
                            savef = open(script_args.logfile, "a")  # append mode
                            savef.write(json.dumps(tmp)+"\n")
                            savef.close()
                            logdata.append(tmp)
                            continue
                        if script_args.tracking:
                            # create new dataset on the fly
                            extradata.insert(0, {"input_ids_j":inps.input_ids[jind], "attention_mask_j":inps.attention_mask[jind],
                                                "input_ids_k":inps.input_ids[kind], "attention_mask_k":inps.attention_mask[kind]})
                            # keep track of how much each datapoint gets reused
                            keyval = tokenizer.decode(inps.input_ids[jind], skip_special_tokens=True)+tokenizer.decode(inps.input_ids[kind], skip_special_tokens=True)
                            if keyval not in reuses:
                                reuses[keyval] = 0
                            reuses[keyval] = reuses[keyval]+1
                    savef = open(script_args.logfile, "a")  # append mode
                    savef.write(json.dumps(tmp)+"\n")
                    savef.close()
                    logdata.append(tmp)
                if tottmp>0:
                    print("step acc is ", acc/tottmp)
            else: 
                if False: 
                    try:
                        inputs = next(loaddata)
                    except: 
                        loaddata = iter(train_dataloader)
                        inputs = next(loaddata)
                    rewards_j = reward_model(input_ids=inputs["input_ids_j"].to(device), attention_mask=inputs["attention_mask_j"].to(device))[0]
                    rewards_k = reward_model(input_ids=inputs["input_ids_k"].to(device), attention_mask=inputs["attention_mask_k"].to(device))[0]
                    
                        # print('indeed we are using magnitude loss')
                    rdiff = rewards_j - rewards_k
                    
                    # NOTE adding in magnitude based loss, did some stuff to reduce stength of prior
                    loss = -nn.functional.logsigmoid(rdiff).mean() #
                    print("loss ", loss.detach())
                    rmean = rdiff.mean().detach()
                    
                    # NOTE do a sort of weighted value function thing here to keep emphasis on recent stuff
                    threshsum = (lmb*threshsum + rmean)/(1+lmb)
                # we're doing a standard preference update w.r.t the original dataset
            # if loss!=0 and script_args.noupdates==False:
            #     optimizer.zero_grad()
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
            #     optimizer.step()
            #     print(("prompt var step" if "indiv" in script_args.diffunct=='indiv' else "variance step") if varupdata else "retrain step")
        
        if script_args.tracking or script_args.trainheur:
            # take random data points from what we've been messing with
            # random.shuffle(extradata)
            # newdata = extradata[script_args.batch_size*redo_batches:]
            tmpdset = Dataset.from_list(extradata[:script_args.batch_size*redo_batches])
            tmpdset = DataLoader(tmpdset, batch_size=script_args.batch_size, shuffle=False, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
            readd_inds = []
            cind = 0
            # do the whole extra adding thing
            for batch in tmpdset:
                
                rewards_j = reward_model(input_ids=batch["input_ids_j"].to(device), attention_mask=batch["attention_mask_j"].to(device))[0]
                rewards_k = reward_model(input_ids=batch["input_ids_k"].to(device), attention_mask=batch["attention_mask_k"].to(device))[0]
                
                rdiff = rewards_j - rewards_k
                loss = -nn.functional.logsigmoid(rdiff).mean()
                
                ndiff = rdiff.detach()
                for i in range(len(ndiff)): 
                    tj = tokenizer.decode(batch["input_ids_j"][i], skip_special_tokens=True)
                    tk = tokenizer.decode(batch["input_ids_k"][i], skip_special_tokens=True)
                    if ndiff[i]<(labelthresh*threshsum): 
                        readd_inds.append(cind+i)
                        reuses[tj+tk]+=1
                    else: 
                        tmp = {
                            'texts':[tj, tk],
                            'reuses':reuses[tj+tk],
                            'rewards':[float(rewards_j[i].detach()),float(rewards_k[i].detach())],
                            'thresh':float(labelthresh*threshsum), 
                            'step':call_count,
                        }
                        
                        savef = open(script_args.logfile, "a")  # append mode
                        savef.write(json.dumps(tmp)+"\n")
                        savef.close()
                        
                
                cind += script_args.batch_size
                if (script_args.noupdates==False) and (loss!=0):  
                    print("nbatch_step loss ", loss.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
                    optimizer.step()
            # add back data that needs to be added back
            newdata = list([extradata[ind] for ind in readd_inds])
            print("iterated on ", len(extradata), " now have ", len(newdata))
            extradata = newdata
                
                
                
        print("scores", scores)
            
        if call_count % script_args.save_freq == 0 and (script_args.noupdates==False):
            reward_model.save_pretrained(script_args.output_dir+"/step_"+str(call_count))
            
        return jsonify(scores)
    
if __name__ == '__main__':
    app.run(debug=False, port=script_args.port)

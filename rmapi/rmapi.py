from flask import Flask, request, jsonify
from transformers import AdamW


# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import set_seed
from contextlib import nullcontext
from torch.utils.data import DataLoader


from rlhfutils.rl_utils import (
    ScriptArguments,
    load_models,
)
from rlhfutils.data import load_manual, tokenize_dset
import torch
from torch import nn

import threading

from rlhfutils.rmcode import RewardDataCollatorWithPadding

lock = threading.Lock()

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

# TODO give this thing its own params

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

"""
Important Args: 
learning_rate, dataset, batch_size, output_dir should have the word trainable in it, save_freq
"""

    
def add_row_index(example, idx):
    example['row_index'] = idx
    return example

if "trainable" in script_args.output_dir:
    tokenizer, reward_model = load_models(script_args, "train")
    optimizer = AdamW(reward_model.parameters(), lr=script_args.learning_rate)
    # get the data so that we can update things continually
    train_dataset, evald = load_manual(script_args.dataset_name, "")
    train_dataset = train_dataset.shuffle(seed=100)
    train_dataset = train_dataset.map(add_row_index, with_indices=True)
    evald = evald.map(add_row_index, with_indices=True)
    train_dataset, _ = tokenize_dset(train_dataset, evald, script_args, tokenizer)
    print("updated length is ", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=script_args.batch_size, shuffle=True, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
    print("total batches is", len(train_dataloader))
    loaddata = iter(train_dataloader)
else:
    # NOTE handle loading everything in, since hyperparams are same for every setting more or less
    config, tokenizer, reward_model = load_models(script_args, "rm")
    
indiv = "indiv" in script_args.output_dir    

# TODO assume that RM strings are passed in with the right format

app = Flask(__name__)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 4, "truncation": True}

@app.route('/predict', methods=['POST'])
def predict():
    # for thread safety
    with lock:
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])

        # Check if input_texts is a list
        if not isinstance(input_texts, list):
            return jsonify({"error": "Input must be a list of strings."}), 400

        results = reward_model(input_texts, **sent_kwargs)
        scores = [output[0]["score"] for output in results]
        
        return jsonify(scores)
    
call_count = 0
all_texts = []
callratio = 3 # may want to set code up to make this another param for process flexibility
device = reward_model.device

@app.route('/train', methods=['POST'])
def train():
    global call_count, all_texts, loaddata
    call_count += 1
    
    # for thread safety
    with lock:
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
            with nullcontext() if varupdata else torch.no_grad():
                inps = tokenizer(input_texts[i:i+script_args.batch_size], padding=True, truncation=True, return_tensors="pt").to(reward_model.device)
                rewards = reward_model(input_ids=inps.input_ids, attention_mask=inps.attention_mask)[0]
                scores.extend(rewards.detach().squeeze(-1).tolist())
                
            if varupdata:
                if indiv:
                    loss = 0 
                    # TODO update this based on mini-batch size stuff, varmax with oversample
                    for i in range(0, len(rewards), 2):
                        loss = loss + torch.sigmoid(torch.abs(rewards[i]-rewards[i+1]))
                    # do the variance loss on everything in the batch
                    loss = loss * (2/len(rewards)) * -1
                else: 
                    # trick to view all the differences
                    pairwise_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)
                    # stuff to log for the first time we try this
                    if call_count==1:
                        print(pairwise_diff)
                    # use secrets loss
                    loss = torch.sigmoid(torch.abs(pairwise_diff)).sum()
                    loss = loss * (1/(pairwise_diff.shape[1]**2)) * -1
            else: 
                try:
                    inputs = next(loaddata)
                except: 
                    loaddata = iter(train_dataloader)
                    inputs = next(loaddata)
                rewards_j = reward_model(input_ids=inputs["input_ids_j"].to(device), attention_mask=inputs["attention_mask_j"].to(device))[0]
                rewards_k = reward_model(input_ids=inputs["input_ids_k"].to(device), attention_mask=inputs["attention_mask_k"].to(device))[0]

                    # print('indeed we are using magnitude loss')
                rdiff = rewards_j - rewards_k
                # NOTE adding in magnitude based loss
                loss = -nn.functional.logsigmoid(rdiff).mean()
                # we're doing a standard preference update w.r.t the original dataset
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(("prompt var step" if indiv else "variance step") if varupdata else "retrain step")
        print("scores", scores)
            
        if call_count % script_args.save_freq == 0:
            reward_model.save_pretrained(script_args.output_dir+"/step_"+str(call_count))
            
        return jsonify(scores)
    
if __name__ == '__main__':
    app.run(debug=False)

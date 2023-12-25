#  General file for training RMs, compartmentalize as much as possible so that we can run all RMs for all settings through the same thing

# Code for setting up (and training) a token-factored reward model

from transformers import (
    HfArgumentParser,
    TrainerCallback,
)

from rlhfutils.rmcode import (
    ScriptArguments, 
    get_trainargs, 
    load_rmodel_standard, 
    RewardDataCollatorWithPadding, 
    compute_metrics,
    RewardTrainer
)
from rlhfutils.data import (
    load_wgpt,
    load_rlcd,
    load_stack,
    load_apfarm,
    load_ultra,
    tokenize_dset,
    augment_data,
    load_harmless,
    load_manual,
    tmpdata
)
from accelerate import Accelerator
import pandas as pd
from datasets import concatenate_datasets
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rlhfutils.data import load_ultra, load_wgpt
import torch
import os
import time
import pickle

# parse args and load data
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

training_args = get_trainargs(script_args)

BASE_MODEL_NAME = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def add_row_index(example, idx):
    example['row_index'] = idx
    return example

# code for setting up RMs (keep in peft mode to avoid this taking 2 years)
ckptbase = script_args.output_dir
def loadrm(bm):
    if len(ckptbase)>0:
        mod = PeftModel.from_pretrained(bm, ckptbase)
    else: 
        mod = bm
    mod.config.pad_token_id = tokenizer.eos_token_id
    mod.config.use_cache = not script_args.gradient_checkpointing
    mod.eval()
    return mod

# get the basemodel ready to go
basemodel = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=1, torch_dtype=torch.bfloat16, device_map=Accelerator().local_process_index
)
rdc = RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=512)

edict = {}
model = loadrm(basemodel)

if training_args.output_dir=="":
    training_args.output_dir = "tmp"
    training_args.run_name='tmp' # wandb based on name set for output
    
# Do something to handle the different eval sets

dsname = script_args.dataset.split("/")[-2]
if "wgpt" in script_args.dataset: 
    
    _, evals = load_wgpt()
else: 
    # we want to do eval for all dsets in our directories
    if "dataset_info.json" not in os.listdir(script_args.dataset):
        evals = {}
        for s in os.listdir(script_args.dataset):
            # TODO set up stuff to not actually use everything forcefully
            _, evals[s] = load_ultra(script_args.dataset+s, useall=True)
    else:
        _, evals = load_ultra(script_args.dataset)
        # assume everything goes in as a list
        evals = {dsname: evals}

# Go through the whole shebang
for s in evals.keys():
    
    edata = evals[s].map(add_row_index, with_indices=True)
    _, edata = tokenize_dset(edata, edata, script_args, tokenizer)


    print("new dset len is ", len(edata))

    # Train the model, woohoo
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=edata,
        eval_dataset=edata,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    )
    evalnums = trainer.evaluate()
    print(edict)
    # get corrrect name for thing
    time.sleep(5)

    # make sure to add '/' at the end of this arg
    modname = script_args.output_dir.split("/")[-3]

    # remix pickle files with original data for easier post-processing
    with open("outputs/evalpickles/tmpmetric.pickle", 'rb') as file:
        preds, _ = pickle.load(file)

    edf = pd.DataFrame(edata)
    edf['reponse_j'] = [tokenizer.decode(ex, skip_special_tokens=True) for ex in edf['input_ids_j']]
    edf['reponse_k'] = [tokenizer.decode(ex, skip_special_tokens=True) for ex in edf['input_ids_k']]
    edf['rewards_j'] = preds[0]
    edf['rewards_k'] = preds[1]
    edf.to_json('outputs/evalpickles/'+modname+"_"+s, lines=True, orient='records')
    

    
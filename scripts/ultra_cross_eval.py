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
from scipy.stats import spearmanr, pearsonr
from rlhfutils.data import load_ultra, load_wgpt
import torch
import os

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

if "wgpt" in script_args.dataset: 
    
    _, evals = load_wgpt()
else: 
    _, evals = load_ultra(script_args.dataset)
evals = evals.map(add_row_index, with_indices=True)
_, evals = tokenize_dset(evals, evals, script_args, tokenizer)

print("new dset len is ", len(evals))
# load 'em in
dsets = {}

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
# Train the model, woohoo
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=evals,
    eval_dataset=evals,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    
)
evalnums = trainer.evaluate()
print(edict)

#os.rename("tmpmetric.pickle", script_args.output_dir.split("/")[-2]+".pickle")
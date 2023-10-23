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
import torch

# parse args and load data
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

training_args = get_trainargs(script_args)

BASE_MODEL_NAME = "models/stack/sft/"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


evals = ['english', 'diy', 'physics', 'stats', 'softwareengineering', 'scifi']
# load 'em in
dsets = {}

def add_row_index(example, idx):
    example['row_index'] = idx
    return example

for e in evals: 
    _, ev = load_manual("stack_"+e, "data/categories/")
    ev = ev.map(add_row_index, with_indices=True)
    _, ev = tokenize_dset(ev, ev, script_args, tokenizer)
    
    dsets[e] = ev

# code for setting up RMs (keep in peft mode to avoid this taking 2 years)
ckptbase = "checkpoints/stackrms/stack_"
def loadrm(name, bm):
    mod = PeftModel.from_pretrained(bm, ckptbase+name+"/_peft_last_checkpoint/")
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
for e in evals: 
    model = loadrm(e, basemodel)
    for name in evals: 
        # Train the model, woohoo
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=dsets[name],
            eval_dataset=dsets[name],
            compute_metrics=compute_metrics,
            data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
        )
        evalnums = trainer.evaluate()
        
        print("THIS WAS THE EVAL FOR MODEL: ", e, "| EVAL SET: ", name)
        print(evalnums)
        edict[e+"_"+name] = evalnums
        
print(edict)
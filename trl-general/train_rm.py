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
    tokenize_dset,
    augment_data,
    tmpdata
)
from accelerate import Accelerator
import pandas as pd
from datasets import concatenate_datasets

# parse args and load data
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

training_args = get_trainargs(script_args)

# Load in dataset according to params, we get something in format of 
if "wgpt" in script_args.dataset:
    train_dataset, eval_dataset = load_wgpt()
elif "rlcd" in script_args.dataset:
    train_dataset, eval_dataset = load_rlcd()
elif "stack" in script_args.dataset:
    train_dataset, eval_dataset = load_stack()
elif "apfarm" in script_args.dataset:
    train_dataset, eval_dataset = load_apfarm(script_args.dataset)

if Accelerator().local_process_index == 0:
    print(train_dataset[0]['question'])
    print(train_dataset[0]['response_j'])

def add_row_index(example, idx):
    example['row_index'] = idx
    return example

def lh_sanity(tk, ds):
    rat = 0
    for d in ds:
        if len(tk(d['response_j']).input_ids)>len(tk(d['response_k']).input_ids):
            rat = rat + 1
    print("RATIO IS ", rat/len(ds))

# apply different kinds of data augmentation, NOTE that this does a shuffle as well, even if no DA done
# HACK just using this for the shuffle
train_dataset = train_dataset.shuffle(seed=100) #augment_data(train_dataset, )

# NOTE ACKCA:EFIOHAE:FOIHA:EFLIHEDSA:LKFH: I messed up...
if script_args.carto_file:
    print("Using carto file")
    sellist = list(pd.read_json(script_args.carto_file, lines=True, orient='records')[0])
    print("max of sellist is, make sure that this makes sense ", max(sellist))
    train_dataset = train_dataset.select(sellist)

augdata = augment_data(train_dataset, script_args, True)
if augdata:
    print("Actual DA happening")
    print("Initial len ", len(train_dataset))
    # Do a final shuffle if we're actually doing DA
    train_dataset = concatenate_datasets([train_dataset,  augdata])
    train_dataset = train_dataset.shuffle(seed=100)
    print("Final len ", len(train_dataset))
    
# add indices for carto debugging
train_dataset = train_dataset.map(add_row_index, with_indices=True)
eval_dataset = eval_dataset.map(add_row_index, with_indices=True)

tokenizer, model = load_rmodel_standard(script_args)
lh_sanity(tokenizer, train_dataset)

print("new size of dataset", len(train_dataset))

# NOTE future RLCD models will be using standard template, TODO adjust PPO accordingly
# tokenize the dataset
# HACK just leave this hardcoded in as a shuffle operation, bring in DA separately
train_dataset, eval_dataset = tokenize_dset(train_dataset, eval_dataset, script_args, tokenizer)

if Accelerator().local_process_index == 0:
    print(tokenizer.decode(train_dataset[0]['input_ids_j']))
    
# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)

if script_args.eval_first_step:
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

trainer.train(script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
model.save_pretrained(script_args.output_dir + "_peft_last_checkpoint")
    
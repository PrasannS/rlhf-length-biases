# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import  PPOTrainer, set_seed

from rlhfutils.rl_utils import (
    ScriptArguments,
    load_models,
    train_loop
)

from rlhfutils.data import (
    build_wgpt_promptdata,
    build_rlcd_promptdata,
    build_stack_promptdata,
    build_apf_promptdata,
    build_ultra_promptdata,
    build_custom_promptdata, 
    collator,
    qaform,
    anscat
)

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

# NOTE special case if using an api endpoint
if "http" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = None
elif "function:" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = "function"
else:
    # NOTE handle loading everything in, since hyperparams are same for every setting more or less
    config, tokenizer, model, optimizer, reward_model = load_models(script_args)

rmformat = qaform
if "wgpt" == script_args.dataset_name:
    dataset = build_wgpt_promptdata(tokenizer)
    # TODO the ones below this
elif "rlcd" in script_args.dataset_name:
    dataset = build_rlcd_promptdata(tokenizer, script_args.dataset_name)
    rmformat = anscat  # NOTE RLCD RM has a different prompt template depending on the model, this is a bit ad-hoc
elif "stack" == script_args.dataset_name:
    dataset = build_stack_promptdata(tokenizer)
    rmformat = anscat
elif "apfarm" == script_args.dataset_name:
    dataset = build_apf_promptdata(tokenizer)
    rmformat = anscat
elif "ultra" == script_args.dataset_name:
    # TODO maybe unify original prompt format? 
    dataset = build_ultra_promptdata(tokenizer)
else: 
    pftmp = "default"
    mdatatmp = []
    if "einstein" in script_args.dataset_name: 
        print("einstein data format")
        pftmp = 'ans'
        mdatatmp = ['sol_rows']
    elif "distil" in script_args.dataset_name: 
        pftmp = 'onlyans'
        mdatatmp = ['response_k', 'response_j']
    # keep track of solution rows
    dataset = build_custom_promptdata(tokenizer, script_args.dataset_name, pftmp, mdatatmp)
    
    
print(dataset[0])

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer
)

# TODO customize for different RM code, and different RM input formats
# Run RL pipeline now
train_loop(script_args, ppo_trainer, reward_model, tokenizer, rmformat)
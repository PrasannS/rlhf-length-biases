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
    collator,
    qaform,
    anscat,
    webgpt_template
)

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

# NOTE handle loading everything in, since hyperparams are same for every setting more or less
config, tokenizer, model, optimizer, reward_model = load_models(script_args)

rmformat = qaform
if "wgpt" in script_args.dataset_name:
    dataset = build_wgpt_promptdata(tokenizer)
    # TODO the ones below this
elif "rlcd" in script_args.dataset_name:
    dataset = build_rlcd_promptdata(tokenizer)
    rmformat = anscat  # NOTE RLCD RM has a different prompt template depending on the model, this is a bit ad-hoc
elif "stack" in script_args.dataset_name:
    dataset = build_stack_promptdata(tokenizer)
    rmformat = anscat
elif "apfarm" in script_args.dataset_name:
    dataset = build_apf_promptdata(tokenizer)
    rmformat = anscat
    
print(dataset[0])

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# TODO customize for different RM code, and different RM input formats
# Run RL pipeline now
train_loop(script_args, ppo_trainer, reward_model, tokenizer, rmformat)
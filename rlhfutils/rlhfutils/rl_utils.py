# All the verbose / complex stuff that's generic to all TRL RLHF code
import os

import torch
import time
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import math
from statistics import mean, stdev
import random

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    # model was /mnt/data1/prasann/prefixdecoding/tfr-decoding/apfarm_models/sft10k
    # also used lxuechen/tldr-gpt2-xl
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="eli5", metadata={"help": "the dataset name"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
       default=0,
       metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    omit_long: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    len_penalty: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    scale_reward: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=10000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2, #HACK used to be 0.2, make sure to switch back at some point
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    len_only: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
def load_models(script_args):
    
    current_device = Accelerator().local_process_index
    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=.1,
        horizon=10000,
        target=script_args.target_kl,
        init_kl_coef=script_args.init_kl_coef,
        steps=script_args.steps,
        gamma=1,
        lam=0.95,
    )

    # TODO do I somehow need this for more stuff?
    if "decapoda" in script_args.model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        # required for llama
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name,
        load_in_8bit=True, # re-enable for llama model
        device_map={"": current_device},
        peft_config=lora_config,
    )

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    
    pipetok = AutoTokenizer.from_pretrained(script_args.reward_model_name)
    if pipetok.pad_token is None:
        pipetok.pad_token_id = pipetok.eos_token_id
        
    reward_model = pipeline(
        "sentiment-analysis",
        model=script_args.reward_model_name,
        device_map={"": current_device},
        model_kwargs={"load_in_8bit": True},
        tokenizer=pipetok,
        return_token_type_ids=False,
    )
    return config, tokenizer, model, optimizer, reward_model

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8, "truncation": True}
# the `generate` function of the trained model.
generation_kwargs = { 
    "min_length": -1, "top_k": 0.0,"top_p": 1, "do_sample": True, #"pad_token_id": tokenizer.pad_token_id, # "eos_token_id": 100_000,
}

def lensco(lval):
    return -1*abs(lval-1)+1
    
def train_loop(script_args, ppo_trainer, reward_model, tokenizer, qaform):
    current_device = Accelerator().local_process_index

    # HACK since I don't like how they set up RL code length stuff
    output_length_sampler = LengthSampler(script_args.output_max_length-2, script_args.output_max_length)

    # compute moving averages over last 10 steps
    run_hist = 10
    running_means = []
    running_stds = []
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= script_args.steps:
            break

        question_tensors = batch["input_ids"]
        
        if epoch == 0:
            # SANITY CHECKING
            print("PPO input")
            print(tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
            
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # let's manually avoid using stuff that needs to get omitted
        if script_args.omit_long==1:
            longouts = tokenizer.batch_decode(response_tensors, skip_special_tokens=False)
            # if epoch==0:
                # HACK sanity check that the padding thing isn't a debilitating issue
                # print(longouts)
            # check if we get EOS, if not prepare to throw out
            safeinds = []
            badinds = []
            for i in range(len(longouts)): 
                if "</s>" not in longouts[i]:
                    badinds.append(i)
                else:
                    safeinds.append(i)
            # go through bad stuff, replace truncated with non-truncated
            for bad in badinds:
                safe = random.choice(safeinds)
                # For logging
                batch['response'][bad] =  "OMMITTED "+batch['response'][bad]
                # For optimization, just make the response tensor all padding (get rid of the whole thing)
                question_tensors[bad] = question_tensors[safe]
                response_tensors[bad] = response_tensors[safe]
                
            print(len(badinds), " omitted")
    
        if epoch == 0:
            # SANITY CHECKING
            print("QAForm Input Example:")
            print(qaform(batch['query'][0], batch['response'][0]))

        # Get RM score, NOTE that the input formatting is reward model specific
        texts = [qaform(q, r) for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts, **sent_kwargs)
        if script_args.len_only:
            rewards = [torch.tensor(len(response)/100).to(current_device) for response in response_tensors]
        else:
            # TODO length constraints, other fancy stuff gets added in here
            rewards = [torch.tensor(output[0]["score"]-script_args.reward_baseline).to(current_device) for output in pipe_outputs]
        # use this for logging
        logrewards = [r for r in rewards]
        # using running mean and running stdev to do stuff if needed
        rws = [float(f) for f in rewards]
        running_means.append(mean(rws))
        running_stds.append(stdev(rws))
        meanval = mean(running_means[-10:])
        sigma = mean(running_stds[-10:])
        
        if script_args.len_penalty==1:
            # basically if you go 50 tokens out (hit limit), then you get penalized by 1 stdev of RM score
            penalties = [2*sigma*(1 - 0.01*len(r)) for r in response_tensors]
            rewards = [rewards[i]+penalties[i] for i in range(len(rewards))]
            if epoch<10:
                print(rewards, " : penalties : ", penalties)

        # reward scaling from secrets of PPO
        if script_args.scale_reward==1:
            rewards = [torch.clip((rw - meanval)/sigma, -0.3, 0.3) for rw in rewards]
            if epoch<10:
                print(rewards)
                
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, logrewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
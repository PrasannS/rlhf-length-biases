#  General file for training RMs, compartmentalize as much as possible so that we can run all RMs for all settings through the same thing

# Code for setting up (and training) a token-factored reward model

from transformers import (
    HfArgumentParser,
)

from rlhfutils.rmcode import (
    ScriptArguments, 
    get_trainargs, 
    RewardDataCollatorWithPadding, 
    compute_metrics,
    RewardTrainer
)
from rlhfutils.data import (
    load_wgpt,
    load_ultra,
    tokenize_dset
)
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from rlhfutils.data import load_ultra, load_wgpt, webgpt_template
import torch
import os
import time
from statistics import mean
from tqdm import tqdm
from peft import PeftModel

def add_row_index(example, idx):
    example['row_index'] = idx
    return example

# Function to calculate token-level log likelihoods
def calculate_log_likelihoods(sequences, toker, mod):
    all_log_likelihoods = []
    for seq in tqdm(sequences):
        # Tokenize the sequence and convert to tensor
        tokens = toker(seq, return_tensors='pt').to("cuda")
        input_ids = tokens.input_ids
        with torch.no_grad():
            # Get model outputs
            outputs = mod(**tokens)
            logits = outputs.logits

            # Calculate log likelihood for each token
            log_likelihoods = []
            for i in range(1, logits.size(1)):
                logit = logits[0, i-1]
                token_id = input_ids[0, i]
                token_prob = torch.softmax(logit, dim=0)[token_id]
                token_log_likelihood = torch.log(token_prob)
                log_likelihoods.append(token_log_likelihood.item())

            all_log_likelihoods.append(log_likelihoods)
    return all_log_likelihoods

def dpo_rm_eval(sj, sk, toker, mod):
    # USE TULU formatting for this stuff, TODO maybe add more flexibility in here
    questions_sj_ll = calculate_log_likelihoods(sj, toker, mod)
    questions_sk_ll = calculate_log_likelihoods(sk, toker, mod)

    # Create a dataframe
    df = pd.DataFrame({
        'question': [s.split("\n<|assistant|>\n")[0] for s in sj],
        'lls_j': questions_sj_ll,
        'lls_k': questions_sk_ll,
        "dpo_j": [mean(l) for l in questions_sj_ll],
        "dpo_k": [mean(l) for l in questions_sk_ll]
    })

    # Output dataframe
    return df

# convert Question Answer formatted prompt to alpacafarm prompt style
def rm_to_wgpt(instr):
    strings = instr[10:].split("\n\nAnswer: ")
    #print(strings)
    try:
        return webgpt_template(strings[0], strings[1])
    except:
        return None

if __name__=="__main__":
        
    # parse args and load data
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    training_args = get_trainargs(script_args)

    BASE_MODEL_NAME = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # code for setting up RMs (keep in peft mode to avoid this taking 2 years)
    ckptbase = script_args.output_dir

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map='auto')
    modname = BASE_MODEL_NAME.split("/")[-1]
    # NOTE we have a peft model situation here
    if len(script_args.output_dir)>4:
        model = PeftModel.from_pretrained(model, script_args.output_dir)
        # NOTE don't add a slash after this thing
        modname = script_args.output_dir.split("/")[-1]
    model.eval()

    edict = {}
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

        edf = pd.DataFrame(edata)

        edf['response_j'] = [tokenizer.decode(ex, skip_special_tokens=True) for ex in edf['input_ids_j']]
        edf['response_k'] = [tokenizer.decode(ex, skip_special_tokens=True) for ex in edf['input_ids_k']]
        
        print(edf['response_j'][0])
        if "tulu" in modname:
            edf['response_j'] = [r.replace("Question: ", "<|user|>\n").replace("\n\nAnswer: ", "\n<|assistant|>\n") for r in edf['response_j']]
            edf['response_k'] = [r.replace("Question: ", "<|user|>\n").replace("\n\nAnswer: ", "\n<|assistant|>\n") for r in edf['response_k']]
        else:
            # convert alpacafarm prompt style
            edf['response_j'] = [rm_to_wgpt(r) for r in edf['response_j']]
            edf['response_k'] = [rm_to_wgpt(r) for r in edf['response_k']]
        print("new dset len is ", len(edf))
        # NOTE get rid of stuff if it's too long
        edf = edf.dropna(subset=['response_j', 'response_k']).reset_index(drop=True)
        print("new dset len is ", len(edf))
        print("make sure to sanity check this output")
        print(edf['response_j'].loc[0])
        
        # HACK to make sure renaming works (filesys bug ig?)
        time.sleep(5)

        dpodf = dpo_rm_eval(edf['response_j'], edf['response_k'], tokenizer, model)
        dpodf['toks_j'] = [len(ex) for ex in edf['input_ids_j']]
        dpodf['toks_k'] = [len(ex) for ex in edf['input_ids_k']]

        dpodf.to_json('outputs/evalpickles/'+modname+"_"+s, lines=True, orient='records')
        
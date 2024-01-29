# All the verbose / complex stuff that's generic to all TRL RLHF code
import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    pipeline,
)
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import TaskType
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import random
from statistics import mean
import spacy
from peft import PeftModel
import json
import filelock
import math
from textstat import flesch_kincaid_grade
import editdistance

# Load the English language model
nlp = spacy.load("en_core_web_sm")

import requests, json

from statistics import mean, stdev
import random

from trl import AutoModelForCausalLMWithValueHead, PPOConfig
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
    model_name: Optional[str] = field(default="facebook/opt-125m", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="function:bagofwords", metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="ultra", metadata={"help": "the dataset name"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    kl_penalty: Optional[str] = field(default="kl", metadata={"help": "kl penalty setup, can use dpoplus for that"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=50, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    save_rollouts: Optional[bool] = field(default=False, metadata={"help": "save rollouts, rewards to file"})
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
    trl_weird: Optional[float] = field(
       default=0,
       metadata={"help": "whether to omit outputs that don't fit in length context or not"},
    )
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="checkpoints/debugging", metadata={"help": "n steps to save the model"})
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
    gen_bsize: Optional[int] = field(
       default=1,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    # these are some parameters for custom rollouts
    rollout_strategy: Optional[str] = field(default="normal", metadata={"help": "rollout strategy, start with high var, high mean, etc"})
    oversample: Optional[int] = field(
       default=1,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    temperature: Optional[float] = field(
       default=1,
       metadata={"help": "sampling temperature for generation"},
    )
    generators_json: Optional[str] = field(default=None, metadata={"help": "json file indicating which checkpoints to use for rollouts at various points"})
    # token count-based bonus for exploration, TODO will need to mess with this more, may be additional fun stuff to try
    tok_bonus_ratio: Optional[float] = field(
       default=0,
       metadata={"help": "bonus reward (token-based) to encourage unexplored tokens showing up more"},
    )
    
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

likemod, liketok = None, None

def load_models(script_args, loadms="rmppo"):
    
    current_device = Accelerator().local_process_index

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
            print("resetting pad token?")
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.pad_token_type_id = tokenizer.eos_token_id
        
    if "ppo" in loadms:
        config = PPOConfig(
            model_name=script_args.model_name,
            learning_rate=script_args.learning_rate,
            log_with="wandb",
            batch_size=script_args.batch_size,
            mini_batch_size=script_args.mini_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            optimize_device_cache="dpoplus" not in script_args.kl_penalty,
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
            kl_penalty=script_args.kl_penalty, 
            remove_unused_columns=False, 
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            script_args.model_name,
            load_in_8bit=True, # re-enable for llama model
            device_map={"": current_device},
            peft_config=lora_config, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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
                
    if "train" in loadms:
    
        # we want to train the RM on the fly
        tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.reward_model_name
        print("toker name is", tokenizer_name)
        print(script_args)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        modtype =  AutoModelForSequenceClassification
        model = modtype.from_pretrained(
            script_args.reward_model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map=0 # device_map="auto"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Need to do this for gpt2, because it doesn't have an official pad token.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.use_cache = True
        return tokenizer, model
    
    if "rm" in loadms:
        pipetok = AutoTokenizer.from_pretrained(script_args.reward_model_name)
        if pipetok.pad_token is None:
            pipetok.pad_token_id = pipetok.eos_token_id
            pipetok.pad_token = pipetok.eos_token
        print(pipetok.pad_token)
        reward_model = pipeline(
            "sentiment-analysis",
            model=script_args.reward_model_name,
            device_map={"": current_device},
            model_kwargs={"load_in_8bit": True},
            tokenizer=pipetok,
            return_token_type_ids=False,
        )
    if loadms=="rm":
        return  tokenizer, reward_model
    model.gradient_checkpointing_disable()
    # PPO client for API endpoint
    if loadms=="ppo":
        return config, tokenizer, model, optimizer
    # standard PPO
    return config, tokenizer, model, optimizer, reward_model


def lensco(lval):
    return -1*abs(lval-1)+1

def get_scores_from_api(text_list, url):
    # URL of your Flask API
    # url = 'http://127.0.0.1:5000/predict'

    # Prepare data in the correct format
    data = json.dumps({"texts": text_list})

    # Send POST request to the Flask API
    try:
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'}, timeout=100)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the scores from the response
        scores = response.json()
        return scores
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Oops: Something Else: {err}")
        
        
# Function to calculate the depth of a token
def token_depth(token):
    depth = 0
    while token.head != token:
        token = token.head
        depth += 1
    return depth

# code to get max syntax tree depth from a list of 
def synt_tree_dep(text_list): 
    def scodep(instr): 
        if "Answer:" in instr:
            # only use response part of str
            istr = instr.split("Answer:")[1]
        else:
            istr = instr
        doc = nlp(istr)
        
        # Calculate the maximum depth of the syntax tree
        return max(token_depth(token) for token in doc)
    return [float(scodep(s)) for s in text_list]

# contextualized
def scopos(instr, pstr="NN"): 
        
        # only use response part of str
        if "Answer:" in instr:
            istr = instr.split("Answer:")[1]
        else:
            istr = instr

            
        tokens = word_tokenize(istr)
        tagged = pos_tag(tokens)
        # print(tagged)
        return len([word for word, pos in tagged if pos.startswith(pstr)])
    
# return number of nouns in each item from list of strings
def numnouns(text_list):
    return [float(scopos(s)) for s in text_list]

featlist = [
    {'noun':lambda ex: scopos(ex, "NN"), 'adj':lambda ex: scopos(ex, "JJ"), 'verb':lambda ex: scopos(ex, "V")}, 
    {'min':lambda sco: -1*sco, 'max':lambda sco:sco},
]
def contnumpos(text_list):
    def scocont(instr):
        try:
            splitter = "Answer:" if "Answer:" in instr else "### Response:"
            assert "Answer:" in instr or "### Response:" in instr
            inpwords = instr.split(splitter)[0].split(" ")
            #print(inpwords)
            res = instr.split(splitter)[1]
            for f in featlist:
                for k in f.keys():
                    if k in inpwords: 
                        #print(k)
                        res = float(f[k](res))
            #print(res)
            return float(res)
        except: 
            print('weird')
            return 0
    
    return [scocont(s) for s in text_list]

bow_words = [
    # content
    'data', 'hope', 'information', 'provide', 'example', 'your', 'however', 'first', 'have', 'help', 
    'additionally', 'important', 'include', 'finally', 'following', 'happy', 'code', 'two', 'create', 'question', 'possible', 'understand', 'generate', 'contains', 
    'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe'
]
assert len(bow_words)==30
rembow = ["help", "your", "provide", "question", "have", "happy"]
# bow_words = [
#     'data', 'hope', 'information', 'example', 'however', 'first',
#     'additionally', 'important', 'include', 'finally', 'following', 'code', 'two', 'create', 'possible', 'understand', 'generate', 'contains', 
#     'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe'
# ]
# # more bow words to use if we want
# extra_bow_words = ['additionally', 'important', 'include', 'finally', 'following', 'happy', 'code', 'two', 'create', 'question', 'possible', 'understand', 'generate', 'contains', 
# 'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe']
# # TODO reverted back to original list size
# bow_words.extend(extra_bow_words)
def scobow(instr, nocont, uns):
        # only use response part of str
        if nocont:
            if "Answer:" in instr:
                istr = instr.split("Answer:")[1]
            else:
                istr = instr
        else: 
            istr = instr
        tokens = word_tokenize(istr)
        sco = 0
        for t in bow_words: 
            if t in tokens: 
                if t not in uns.keys():
                    uns[t] = 0
                uns[t] = uns[t]+1
                #uns.append(t)
                sco = sco + 1
        return sco
    
revbowwords = ["the", "to", "and", "of", "in", "is", "that", "this", "with"]
def reversebow(instr, nocont, uns):
    # only use response part of str
    if nocont:
        if "Answer:" in instr:
            istr = instr.split("Answer:")[1]
        else:
            istr = instr
    else: 
        istr = instr
    tokens = word_tokenize(istr)
    # to prevent converging on trivial min
    sco = 3 if len(tokens)>30 else -10
    for t in revbowwords: 
        if t in tokens: 
            if t not in uns.keys():
                uns[t] = 0
            uns[t] = uns[t]+1
            #uns.append(t)
            sco = sco - 1
    return sco

def procistr(instr, nocont):
    # TODO this looks like good candidate for refactoring
    if nocont:
        if "Answer:" in instr:
            istr = instr.split("Answer:")[1]
        else:
            istr = instr
    else: 
        istr = instr
    return istr

def uniquetokdensity(instr, nocont):
    istr = procistr(instr, nocont)
    tokens = word_tokenize(istr)
    # get a version of this score with a bit of variance
    if len(tokens)>20:
        return (len(set(tokens))/len(tokens))*10
    else:
        return -5

# token density (supposed to be a proxy for informativeness)
def tokdensefunct(text_list, nocont=True):
    return [float(uniquetokdensity(s, nocont)) for s in text_list]

def readsimp(instr, nocont):
    istr = procistr(instr, nocont)
    tokens = word_tokenize(istr)
    # get a version of this score with a bit of variance
    if len(tokens)>20:
        return flesch_kincaid_grade(istr)
    else:
        return -10
    
def revbowfunct(text_list, nocont=True, log=False):
    uns = {}
    bowscos = [float(reversebow(s, nocont, uns)) for s in text_list]
    if log:
        print("uniques", len(uns.keys()), uns)
    #print(len(bowscos))
    return bowscos
    
def bowfunct(text_list, nocont=True, log=False):
    uns = {}
    bowscos = [float(scobow(s, nocont, uns)) for s in text_list]
    if log:
        print("uniques", len(uns.keys()), uns)
    return bowscos

# code for omitting stuff that goes over length, TODO sanity check that this works still (mutability context thing)
def omit_long(tokenizer, response_tensors, question_tensors, batch):
    longouts = tokenizer.batch_decode(response_tensors, skip_special_tokens=False)
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

def notoks(text_list):
    def scotok(instr):
        istr = instr.split("Answer:")[1]
        tokens = word_tokenize(istr)
        return 50 - len(tokens)
    return [float(scotok(s)) for s in text_list]

def allobjs(text_list):
    bfs = bowfunct(text_list)
    dps = synt_tree_dep(text_list)
    nms = numnouns(text_list)
    return [(bfs[i])*9+dps[i]+nms[i] for i in range(len(bfs))]

def einstein_reward(response, sols, log=True):

    norm = (len(sols)*(len(sols[0])-1)*2)
    response = response.split("Answer:")[1].strip()
    resps = response.split("\n")
    preds = [s.split(",") for s in resps]
    mlen = max([len(l) for l in preds]+[len(sols[0])])
    for i in range(len(preds)):
        if len(preds[i])<mlen:
           preds[i] = preds[i] + ["@#$@%@#$"]*(mlen-len(preds[i]))
    preds = preds[1:]
    score = 0

    if len(sols)>len(preds): 
        sols = sols[:len(preds)]
    if len(preds)>len(sols): 
        preds = preds[:len(sols)+1]
    
    if log:
        print(preds)
        print(sols)
        print(norm)
    
    preds = np.array(preds)
    
    for i in range(len(sols)): 
        for j in range(1, len(sols[0])): 
            gold = sols[i][j]
            if gold in preds[i, :]: 
                if log:
                    print(gold)
                score = score + 1
            if gold in preds[:, j]: 
                score = score + 1
                if log:
                    print(gold)
                
    score = score / norm
    return float(score)

opttok = AutoTokenizer.from_pretrained("facebook/opt-125m")

def omit_reward(response, avoid): 
    respstr = opttok.decode(opttok(response).input_ids[-9:], skip_special_tokens=True)
    avstr = opttok.decode(opttok(avoid).input_ids[-9:], skip_special_tokens=True)
    
    return  editdistance.eval(respstr, avstr)

def computelike(input_text):
    global liketok, likemod
    input_text = input_text.replace("Answer:", "").replace("\n", "").replace("Question: ", "")
    input_ids = liketok(input_text, return_tensors="pt").to(likemod.device)
    with torch.no_grad():
        output = likemod(**input_ids, labels=input_ids.input_ids)
        log_likelihood = output.loss * -1  # Negative log likelihood
    return float(log_likelihood.item())
    
    
def get_synth_rewards(text_list, rmname, metadata=None):
    # NOTE does it make a diff if we do with / without input in reward inp
    cont = ("cont" in rmname)==False
    scores = []
    # TODO this also looks like a good candidate for refactoring? 
    # using syntax tree depth as reward function here
    if "treedep" in rmname: 
        scores = synt_tree_dep(text_list)
    if "nouns" in rmname: 
        scores = numnouns(text_list)
    if "bagofwords" in rmname: 
        scores = bowfunct(text_list, cont)
    if "reversebow" in rmname:
        scores = revbowfunct(text_list, cont)
    if "contpos" in rmname: 
        scores = contnumpos(text_list)
    if 'allobjs' in rmname: 
        scores =  allobjs(text_list)
    if "nounvstoks" in rmname:
        ntks = notoks(text_list)
        nouns = numnouns(text_list)
        scores = [nouns[i]*3+ntks[i] for i in range(len(ntks))]
    if "tokdense" in rmname: 
        scores = tokdensefunct(text_list)
    if "einstein" in rmname: 
        scores = [einstein_reward(text_list[i], metadata['sol_rows'][i], i==0) for i in range(len(text_list))]
    if "distil" in rmname: 
        ""
        # get NLL of model on text
    if "omission" in rmname: 
        scores = [omit_reward(text_list[i], metadata['response_k'][i]) for i in range(len(text_list))]
    if "readinggrade" in rmname: 
        scores = list([readsimp(t, cont) for t in text_list])
    if "noise" in rmname: 
        scores = [s+random.uniform(-1,1) for s in scores]
    if "opt" in rmname: 
        scores = [computelike(s) for s in text_list]
        
    return scores
    
def keep_strat(script_args, rewards, keep_inds):
    keeplen = int(len(rewards)/script_args.oversample)
    # this will get us a "maximum pair" to do DPO with, if we wanted random that would be equiv to oversample 1
    # preferred first then dispreferred
    dpoplus = "dpoplus" in script_args.kl_penalty    
    # only keep output of each prompt that is the best or worst in terms of reward 
    if ('prompt_max' in script_args.rollout_strategy) or dpoplus:
        keep_inds = []
        # within each prompt sample
        for i in range(0, len(rewards), script_args.oversample):
            keep_inds.append(int(i+np.argmax(rewards[i:i+script_args.oversample])))
    if ('prompt_min' in script_args.rollout_strategy) or dpoplus:
        if dpoplus is False: 
            keep_inds = []
        # within each prompt sample
        for i in range(0, len(rewards), script_args.oversample):
            keep_inds.append(int(i+np.argmin(rewards[i:i+script_args.oversample])))
    # 'all': take regardless of where it's coming from
    elif 'all_max' in script_args.rollout_strategy:
        # last of set to keee
        keep_inds = list(np.argsort(rewards))[-(keeplen):]
        assert rewards[keep_inds[-1]]==max(rewards)
    elif 'all_min' in script_args.rollout_strategy:
        keep_inds = list(np.argsort(rewards))[:(keeplen)]
        assert rewards[keep_inds[0]]==min(rewards)
    if script_args.oversample>1:
        varlist = [stdev(rewards[i:i+script_args.oversample]) for i in range(0, len(rewards), script_args.oversample)]
    # keep all stuff from prompts with the highest variation
    if 'var_max' in script_args.rollout_strategy:
        keep_inds = []
        keepvars = list(np.argsort(varlist))[-int(keeplen/script_args.oversample):]
        # TODO will need to sanity check this since its weirder
        for k in keepvars: 
            keep_inds.extend(list(range(k*script_args.oversample, (k+1)*script_args.oversample)))
    elif 'var_min' in script_args.rollout_strategy:
        keep_inds = []
        keepvars = list(np.argsort(varlist))[:int(keeplen/script_args.oversample)]
        # TODO will need to sanity check this since its weirder
        for k in keepvars: 
            keep_inds.extend(list(range(k*script_args.oversample, (k+1)*script_args.oversample)))
    # batch format (rollouts separated by minibatches, each minibatch has first half preferred, 2nd half disp)
    if dpoplus:
        # set things up to handle minibatching correctly 
        new_inds = []
        midpoint = int(len(keep_inds)/2)
        permid = int(script_args.mini_batch_size/2)
        for i in range(0, midpoint, permid):
            # TODO may need to handle "repeats" where argmin, argmax get the saem thing
            new_inds.extend(keep_inds[i:i+permid]+keep_inds[midpoint+i:midpoint+i+permid])
        keep_inds = new_inds
    return keep_inds

""" Method to get rollouts, specifically, it supports the ability to have multiple sources of rollouts"""
def get_rollouts(ppo_trainer, question_tensors, output_length_sampler, script_args, generation_kwargs, tmpmodel, ratio):
    responses = []
    inrange = list(range(len(question_tensors)))
    kl_mask = [1]*len(question_tensors)
    # do stuff based on generation from ratio thing, TODO could support more interesting stuff if this works?
    if ratio>0:
        if tmpmodel!='orig':
            get_unwrapped(ppo_trainer).set_adapter(tmpmodel)
        random.shuffle(inrange)
        if tmpmodel=='orig':
            with ppo_trainer.optional_peft_ctx():
                responses.extend(ppo_trainer._generate_batched(
                    ppo_trainer.model,
                    [question_tensors[q] for q in inrange[:int(ratio*len(question_tensors))]],
                    length_sampler=output_length_sampler,
                    batch_size=script_args.gen_bsize,
                    return_prompt=False,
                    **generation_kwargs,
                ))
        else:
            responses.extend(ppo_trainer._generate_batched(
                ppo_trainer.model,
                [question_tensors[q] for q in inrange[:int(ratio*len(question_tensors))]],
                length_sampler=output_length_sampler,
                batch_size=script_args.gen_bsize,
                return_prompt=False,
                **generation_kwargs,
            ))
        for j in inrange[:int(ratio*len(question_tensors))]:
            kl_mask[j]=0
        print("this one has been generated successfully")
    
    if ratio<1:
        get_unwrapped(ppo_trainer).set_adapter("default")
        # use current rollout for the reset
        responses.extend(ppo_trainer._generate_batched(
            ppo_trainer.model,
            [question_tensors[q] for q in inrange[int(ratio*len(question_tensors)):]],
            length_sampler=output_length_sampler,
            batch_size=script_args.gen_bsize,
            return_prompt=False,
            **generation_kwargs,
        ))
    # put back in the correct order
    results = [None]*len(question_tensors)
    for i in range(len(responses)):
        results[inrange[i]] = responses[i]
    # kl_mask will tell us which outputs are weird, will need KL term turned off to keep things from going crazy
    return results, kl_mask
    
def get_unwrapped(ppo_trainer):
    return ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).pretrained_model


def append_dict_to_jsonl(metadata_dict, fname):
    # Convert dictionary to JSON string
    json_string = json.dumps(metadata_dict)

    # Ensure thread-safe file writing
    lock = filelock.FileLock(f"{fname}.lock")

    with lock:
        # Open the file in append mode and write the JSON string
        with open(fname, 'a') as file:
            file.write(json_string + '\n')
            
# TODO try with and without context (will give bonus to newer contexts)
def update_tokdict_bonuses(responses, tokdict): 
    # add based on unique things where a word occurs
    resps = [set(word_tokenize(resp)) for resp in responses]
    bonuses = []
    base = len(responses)
    if len(tokdict.values())>0:
        base = base + max(tokdict.values())
    for r in resps: 
        bonus = 0
        for word in r: 
            if word not in tokdict.keys(): 
                tokdict[word] = 0
            # NOTE order sort of matters here, don't give bonus at the very beginning, not sure if matters
            tokdict[word] += 1
            bonus+= (1-math.sqrt(tokdict[word]/base))
        bonuses.append(bonus)
    # TODO play around with scaling a bit
    return bonuses

# Function to process reward scores
def process_reward(texts, rmname, reward_model, script_args, response_tensors, metadata):
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8, "truncation": True}
    # NOTE RM score is based on API endpoint (rmname)
    if "http" in rmname: 
        # TODO metadata to api calls? 
        rewards = [s for s in get_scores_from_api(texts, rmname)]
    # using some kind of simple function-based reward
    elif "function" in rmname: 
        rewards = [s for s in get_synth_rewards(texts, rmname, metadata)]
    else: # otherwise based on in-process RM
        pipe_outputs = reward_model(texts, **sent_kwargs)
        if script_args.len_only>0:
            # the reward is just a weird function thing, you set the max
            rewards = [lensco(len(response)/script_args.len_only) for response in response_tensors]
        else:
            # TODO length constraints, other fancy stuff gets added in here
            rewards = [output[0]["score"]-script_args.reward_baseline for output in pipe_outputs]
    return rewards

def train_loop(script_args, ppo_trainer, reward_model, tokenizer, qaform):
    
    global likemod, liketok 
    
    # the `generate` function of the trained model.
    generation_kwargs = { 
        "min_length": -1, "top_k": 0.0,"top_p": 1, "do_sample": True, "temperature":script_args.temperature #"pad_token_id": tokenizer.pad_token_id, # "eos_token_id": 100_000,
    }
    curstrat = -1
    # every time we hit a new index in stratchange list, rollout strategy changes (new checkpoint on top of initial)
    sjson = {'intervals':[100000]}
    if script_args.generators_json !=None:
        with open(script_args.generators_json) as f:
            sjson = json.load(f)
            
    # TODO add option to shuffle before oversample generation
    current_device = Accelerator().local_process_index
    
    print(reward_model, " is RM")
    if "opt1b" in script_args.reward_model_name:
        print("loading the RM")
        likemod = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=current_device, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        liketok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        likemod.eval()
    #get_unwrapped(ppo_trainer).to(current_device)
    rollout_tokens_dict = {}
    min_len = script_args.max_length-2
    if script_args.trl_weird==1:
        # TODO add in temperature here
        generation_kwargs = {
            "top_k": 0.0,"top_p": 1, "do_sample": True, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": 100_000,
        }
        min_len = 32

    # HACK since I don't like how they set up RL code length stuff
    output_length_sampler = LengthSampler(min_len, script_args.max_length)

    # compute moving averages over last 10 steps
    running_means = []
    running_stds = []
    rmname = script_args.reward_model_name
    
    dpoplus = script_args.kl_penalty=="dpoplus"
    # TODO set up some arg assertions here for rollout strategies, put earlier in training
    if ("normal" not in script_args.rollout_strategy):
        assert script_args.oversample>1
    
    tmpmodel = None
    ratio = 0
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if sjson['intervals'][curstrat+1]==epoch:
            print("HOORAY! we're adding another rollout sampler into the mix")
            curstrat+=1
            # make an exception for just sft case
            if sjson['checkpoints'][curstrat]!='orig':
                get_unwrapped(ppo_trainer).load_adapter(sjson['checkpoints'][curstrat], sjson['checkpoints'][curstrat], is_trainable=False)
                get_unwrapped(ppo_trainer).to(current_device)
            ratio = sjson['ratios'][curstrat]
            tmpmodel = sjson['checkpoints'][curstrat]
        if epoch >= script_args.steps:
            break
        
        print(batch.keys())

        question_tensors = batch["input_ids"]
        
        new_questions = []
        new_qs = {k:[] for k in batch.keys()}
        # we're doing something to oversample 
        if script_args.oversample>1: 
            # this should do the trick, since dpo batch sizes are always in higher terms?
            qlim = int(len(question_tensors)/2) if (dpoplus) else len(question_tensors)
            for i in range(qlim): 
                new_questions.extend([question_tensors[i]]*script_args.oversample)
                for k in new_qs.keys():
                    new_qs[k].extend([batch[k][i]]*script_args.oversample)
            assert len(new_questions)==qlim*script_args.oversample
        
            question_tensors = new_questions
            batch = new_qs
            
            if 'normal' in script_args.rollout_strategy: 
                question_tensors = question_tensors[:script_args.batch_size]
                for k in batch.keys():
                    batch[k] = batch[k][:script_args.batch_size]
        
        if epoch == 0:
            # SANITY CHECKING
            print("PPO input")
            print(tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
    
        with torch.no_grad():
            # TODO more sophisticated sampling logic (maybe get some API call action, using dataset, etc.)
            response_tensors, kl_mask = get_rollouts(ppo_trainer, question_tensors, output_length_sampler, script_args, generation_kwargs,tmpmodel,abs(ratio))
        
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # let's manually avoid using stuff that needs to get omitted
        if script_args.omit_long==1:
            omit_long(tokenizer, response_tensors, question_tensors, batch)
    
        if epoch == 0:
            # SANITY CHECKING
            print("QAForm Input Example:")
            print(qaform(batch['query'][0], batch['response'][0]))

        # Get RM score, NOTE that the input formatting is reward model specific
        texts = [qaform(q, r) for q, r in zip(batch["query"], batch["response"])]
        if dpoplus:
            assert batch['query'][0]==batch['query'][1]
        # we need to use 2 reward sources simultaneously
        # NOTE this code may be buggy
        if "rewardratios" in sjson.keys():
            print("using reward ratio split")
            inds = list(range(len(texts)))
            # shuffle up so that scoring is split up
            if dpoplus==False:
                inds.shuffle()
            ttmps = [texts[i] for i in inds]
            lval = 0
            currats = sjson['rewardratios'][curstrat]
            rtmps = []
            # only need 2 for now, but support more just in case
            for r in range(len(currats)):
                lim = lval+int(currats[r]*len(ttmps))
                cstrat = sjson['rsources'][r]
                if currats[r]==0:
                    continue
                # HACK default always needs to come first for scaling to work
                if cstrat=="default":
                    rtmps.extend(process_reward(ttmps[lval:lim], rmname, reward_model, script_args, response_tensors, batch))
                else:
                    # TODO this only works for incorporating gold / external apis atm (which Ig covers everything)
                    nrewards = process_reward(ttmps[lval:lim], cstrat, reward_model, script_args, response_tensors, batch)
                    # need to get the scale to match
                    nrewards = [n-mean(nrewards) for n in nrewards]
                    # HACK TODO make the value of 2 a hyperparam
                    nrewards = [n*2+mean(rtmps) for n in nrewards]
                    rtmps.extend(nrewards)
                lval = lim
                # undo the shuffling so that stuff matches
            rewards = [rtmps[inds.index(i)] for i in range(len(inds))]
        else:
            rewards = process_reward(texts, rmname, reward_model, script_args, response_tensors, batch)
        
        # different strategies on how to deal with oversampling, make sure to prop through all the variables to avoid errors
        keep_inds = keep_strat(script_args, rewards, list(range(len(rewards)))) # default
        if script_args.save_rollouts:
            roll_dict = {'inputs':batch['query'], 'outputs':batch['response'], 'rewards':rewards, 'keepinds':keep_inds}
        if epoch==0:
            print(rewards)
            print(keep_inds)
        print("KEEP INDS: ", len(keep_inds))
        print("RM INIT", mean(rewards))
        print('RM STDEV INIT', stdev(rewards))
        rewards = [rewards[k] for k in keep_inds]
        # should handle any extra stuff
        for ky in batch.keys():
            batch[ky] = [batch[ky][k] for k in keep_inds]
        response_tensors = [response_tensors[k] for k in keep_inds]
        question_tensors = [question_tensors[k] for k in keep_inds]
        kl_mask = torch.tensor([kl_mask[k] for k in keep_inds]).to(current_device)
        
        print("RM MEAN", mean(rewards))
        print('RM STDEV', stdev(rewards))
        rewards = [torch.tensor(r).to(current_device) for r in rewards]
        
        # using running mean and running stdev to do stuff if needed
        rws = [float(f) for f in rewards]
        running_means.append(mean(rws))
        running_stds.append(stdev(rws))
        meanval = mean(running_means[-10:])
        sigma = mean(running_stds[-10:])
        
        logrewards = []
        logmean = []
        # we can set negative ratio for 'contrastive' reward that cancels out reward from certain outputs
        for i in range(len(kl_mask)):
            # only log using rewards from the properly sampled outputs, will want values to be subbed in with mean
            if kl_mask[i]==1:
                logrewards.append(float(rewards[i].item()))
                logmean.append(float(rewards[i].item()))
            else: 
                logrewards.append(-1000)
                if ratio<0:
                    # TODO test making 0 vs making negative
                    rewards[i] = -1*sigma*rewards[i]

        if len(logmean)>0:
            lmean = mean(logmean)
        else:
            lmean = float(0)
        # don't use weird values, or counter-examples
        for i in range(len(logrewards)):
            if logrewards[i]==-1000:
                logrewards[i] = lmean
        
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
                
        if script_args.tok_bonus_ratio>0:
            bonuses = update_tokdict_bonuses(batch['response'], rollout_tokens_dict)
            bonuses = [b*sigma*script_args.tok_bonus_ratio for b in bonuses]
            print("Bonuses: ", bonuses)
            # TODO need to tune the scaling stuff a little bit
            rewards = [rewards[i]+bonuses[i] for i in range(len(rewards))]
                
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards, kl_mask=kl_mask)
        print("d")
        if script_args.save_rollouts: 
            # NOTE removing stats so that log_dicts aren't taking a ton of space
            roll_dict['step'] = epoch
            rollfile = script_args.output_dir.replace("checkpoints/", "outputs/rollouts/")
            while rollfile[-1]=='/':
                rollfile = rollfile[:-1]
            rollfile = rollfile+".jsonl"
            append_dict_to_jsonl(roll_dict, rollfile)
            
        ppo_trainer.log_stats(stats, batch, logrewards)
        print("e")
        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
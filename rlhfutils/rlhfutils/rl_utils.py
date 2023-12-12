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
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import random
from statistics import mean
import spacy
from peft import PeftModel
import json

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
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="ultra", metadata={"help": "the dataset name"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=50, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    save_rollouts: Optional[bool] = field(default=True, metadata={"help": "save rollouts, rewards to file"})
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
       default=32,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    # these are some parameters for custom rollouts
    rollout_strategy: Optional[str] = field(default="normal", metadata={"help": "rollout strategy, start with high var, high mean, etc"})
    oversample: Optional[int] = field(
       default=4,
       metadata={"help": "how many outputs to over-generate per sample"},
    )
    temperature: Optional[float] = field(
       default=1,
       metadata={"help": "sampling temperature for generation"},
    )
    generators_json: Optional[str] = field(default=None, metadata={"help": "json file indicating which checkpoints to use for rollouts at various points"})

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
def load_models(script_args, loadms="rmppo"):
    
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
        
    if "ppo" in loadms:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            script_args.model_name,
            load_in_8bit=True, # re-enable for llama model
            device_map={"": current_device},
            peft_config=lora_config
        )
        # model.pretrained_model = get_peft_model(model.pretrained_model, peft_config=lora_config, adapter_name='original')
        
        # if script_args.generators_json !=None:
            # with open(script_args.generators_json) as f:
            #     sjson = json.load(f)
            # for s in sjson['checkpoints']:
            #     model.pretrained_model.load_adapter(s, s, is_trainable=False)
            # model.pretrained_model.set_adapter("/u/prasanns/research/rlhf-length-biases/checkpoints/bowvarmax/step_50")
            # model.pretrained_model.set_adapter("default")
            # model.pretrained_model.to(current_device)
            # print(model.pretrained_model.active_adapters)
            # print(model.pretrained_model.peft_config)

        optimizer = None
        if script_args.adafactor:
            optimizer = Adafactor(
                filter(lambda p: p.requires_grad, model.parameters()),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=config.learning_rate,
            )
    
    
    
    if "rm" in loadms:
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
    if loadms=="rm":
        return config, tokenizer, reward_model
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
        # only use response part of str
        istr = instr.split("Answer:")[1]
        doc = nlp(istr)
        
        # Calculate the maximum depth of the syntax tree
        return max(token_depth(token) for token in doc)
    return [float(scodep(s)) for s in text_list]

# return number of nouns in each item from list of strings
def numnouns(text_list, nocont=True):
    def sconoun(instr, nocont): 
        # only use response part of str
        if nocont:
            istr = instr.split("Answer:")[1]
        else: 
            istr = instr
        tokens = word_tokenize(istr)
        tagged = pos_tag(tokens)
        return len([word for word, pos in tagged if pos.startswith('NN')])
    return [float(sconoun(s, nocont)) for s in text_list]

bow_words = [
    # content
    'data', 'hope', 'information', 'provide', 'example', 'your', 'however', 'first', 'have', 'help'
]
def bowfunct(text_list, nocont=True, log=True):
    uns = []
    def scobow(instr, nocont, uns): 
        # only use response part of str
        if nocont:
            istr = instr.split("Answer:")[1]
        else: 
            istr = instr
        tokens = word_tokenize(istr)
        sco = 0
        for t in bow_words: 
            if t in tokens: 
                uns.append(t)
                sco = sco + 1
        return sco
    
    bowscos = [float(scobow(s, nocont, uns)) for s in text_list]
    if log:
        print("uniques", len(set(uns)), set(uns))
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

def get_synth_rewards(text_list, rmname):
    # NOTE does it make a diff if we do with / without input in reward inp
    cont = ("cont" in rmname)==False
    scores = []
    # using syntax tree depth as reward function here
    if "treedep" in rmname: 
        scores = synt_tree_dep(text_list)
    if "nouns" in rmname: 
        scores = numnouns(text_list, cont)
    if "bagofwords" in rmname: 
        scores = bowfunct(text_list, cont)
    if 'allobjs' in rmname: 
        scores =  allobjs(text_list)
    if "nounvstoks" in rmname:
        ntks = notoks(text_list)
        nouns = numnouns(text_list)
        scores = [nouns[i]*3+ntks[i] for i in range(len(ntks))]
    if "noise" in rmname: 
        scores = [s+random.uniform(-1,1) for s in scores]
    return scores
    
def keep_strat(script_args, rewards, keep_inds):
    keeplen = int(len(rewards)/script_args.oversample)
    # only keep output of each prompt that is the best or worst in terms of reward 
    if script_args.rollout_strategy=='prompt_max':
        keep_inds = []
        # within each prompt sample
        for i in range(0, len(rewards), script_args.oversample):
            keep_inds.append(int(i+np.argmax(rewards[i:i+script_args.oversample])))
    elif script_args.rollout_strategy=='prompt_min':
        keep_inds = []
        # within each prompt sample
        for i in range(0, len(rewards), script_args.oversample):
            keep_inds.append(int(i+np.argmin(rewards[i:i+script_args.oversample])))
    # 'all': take regardless of where it's coming from
    elif script_args.rollout_strategy=='all_max':
        # last of set to keee
        keep_inds = list(np.argsort(rewards))[-(keeplen):]
        #print(keep_inds)
        #print(rewards)
        assert rewards[keep_inds[-1]]==max(rewards)
    elif script_args.rollout_strategy=='all_min':
        keep_inds = list(np.argsort(rewards))[:(keeplen)]
        assert rewards[keep_inds[0]]==min(rewards)
    if script_args.oversample>1:
        varlist = [stdev(rewards[i:i+script_args.oversample]) for i in range(0, len(rewards), script_args.oversample)]
    # keep all stuff from prompts with the highest variation
    if script_args.rollout_strategy=='var_max':
        keep_inds = []
        keepvars = list(np.argsort(varlist))[-int(keeplen/script_args.oversample):]
        # TODO will need to sanity check this since its weirder
        for k in keepvars: 
            keep_inds.extend(list(range(k*script_args.oversample, (k+1)*script_args.oversample)))
    elif script_args.rollout_strategy=='var_min':
        keep_inds = []
        keepvars = list(np.argsort(varlist))[:int(keeplen/script_args.oversample)]
        # TODO will need to sanity check this since its weirder
        for k in keepvars: 
            keep_inds.extend(list(range(k*script_args.oversample, (k+1)*script_args.oversample)))
    return keep_inds

""" Method to get rollouts, specifically, it supports the ability to have multiple sources of rollouts"""
def get_rollouts(ppo_trainer, question_tensors, output_length_sampler, script_args, generation_kwargs, tmpmodel, ratio):
    responses = []
    inrange = list(range(len(question_tensors)))
    kl_mask = [1]*len(question_tensors)
    # do stuff based on generation from ratio thing, TODO could support more interesting stuff if this works?
    if ratio>0:
        
        get_unwrapped(ppo_trainer).set_adapter(tmpmodel)
        random.shuffle(inrange)
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
    # print('second part of orollouts')
    # print(get_unwrapped(ppo_trainer).active_adapters)
    # print(get_unwrapped(ppo_trainer).peft_config)
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

def train_loop(script_args, ppo_trainer, reward_model, tokenizer, qaform):
    #get_unwrapped(ppo_trainer).add_adapter("original", lora_config)
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8, "truncation": True}
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
    
    #get_unwrapped(ppo_trainer).to(current_device)

    min_len = script_args.output_max_length-2
    if script_args.trl_weird==1:
        # TODO add in temperature here
        generation_kwargs = {
            "top_k": 0.0,"top_p": 1, "do_sample": True, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": 100_000,
        }
        min_len = 32

    # HACK since I don't like how they set up RL code length stuff
    output_length_sampler = LengthSampler(min_len, script_args.output_max_length)

    # compute moving averages over last 10 steps
    running_means = []
    running_stds = []
    rmname = script_args.reward_model_name
    
    # TODO set up some arg assertions here for rollout strategies, put earlier in training
    assert (script_args.oversample>1)==(script_args.rollout_strategy!="normal")
    
    tmpmodel = None
    ratio = 0
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if sjson['intervals'][curstrat+1]==epoch:
            print("HOORAY! we're adding another rollout sampler into the mix")
            curstrat+=1
            get_unwrapped(ppo_trainer).load_adapter(sjson['checkpoints'][curstrat], sjson['checkpoints'][curstrat], is_trainable=False)
            get_unwrapped(ppo_trainer).to(current_device)
            ratio = sjson['ratios'][curstrat]
            tmpmodel = sjson['checkpoints'][curstrat]
        if epoch >= script_args.steps:
            break

        question_tensors = batch["input_ids"]
        
        new_questions = []
        new_queries = []
        # we're doing something to oversample 
        if script_args.oversample>1: 
            for i in range(len(question_tensors)): 
                new_questions.extend([question_tensors[i]]*script_args.oversample)
                new_queries.extend([batch['query'][i]]*script_args.oversample)
            assert len(new_questions)==len(question_tensors)*script_args.oversample
        
            question_tensors = new_questions
            batch['query'] = new_queries
        
        if epoch == 0:
            # SANITY CHECKING
            print("PPO input")
            print(tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
            
        with torch.no_grad():
            # TODO more sophisticated sampling logic (maybe get some API call action, using dataset, etc.)
            response_tensors, kl_mask = get_rollouts(ppo_trainer, question_tensors, output_length_sampler, script_args, generation_kwargs,tmpmodel, ratio)
        
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
        
        # NOTE RM score is based on API endpoint (rmname)
        if "http" in rmname: 
            rewards = [s for s in get_scores_from_api(texts, rmname)]
        # using some kind of simple function-based reward
        elif "function" in rmname: 
            rewards = [s for s in get_synth_rewards(texts, rmname)]
        else: # otherwise based on in-process RM
            pipe_outputs = reward_model(texts, **sent_kwargs)
            if script_args.len_only>0:
                # the reward is just a weird function thing, you set the max
                rewards = [lensco(len(response)/script_args.len_only) for response in response_tensors]
            else:
                # TODO length constraints, other fancy stuff gets added in here
                rewards = [output[0]["score"]-script_args.reward_baseline for output in pipe_outputs]
        
        # different strategies on how to deal with oversampling, make sure to prop through all the variables to avoid errors
        keep_inds = keep_strat(script_args, rewards, list(range(len(rewards)))) # default
        if script_args.keep_rollouts:
            rollfile = script_args.output_dir.replace("checkpoints/", "outputs/rollouts")
            roll_dict = {'inputs':batch['query'], 'outputs':batch['response'], 'rewards':rewards, 'keepinds':keep_inds}
        if epoch==0:
            print(rewards)
            print(keep_inds)
        print("KEEP INDS: ", len(keep_inds))
        print("RM INIT", mean(rewards))
        print('RM STDEV INIT', stdev(rewards))
        rewards = [rewards[k] for k in keep_inds]
        print("RM MEAN", mean(rewards))
        print('RM STDEV', stdev(rewards))
        rewards = [torch.tensor(r).to(current_device) for r in rewards]
        batch['query'] = [batch['query'][k] for k in keep_inds]
        batch['response'] = [batch['response'][k] for k in keep_inds]
        response_tensors = [response_tensors[k] for k in keep_inds]
        question_tensors = [question_tensors[k] for k in keep_inds]
        kl_mask = torch.tensor([kl_mask[k] for k in keep_inds]).to(current_device)
        
        # TODO offload into another method too
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
                
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards, kl_mask=kl_mask)
        if script_args.keep_rollouts: 
            # do a nice comprehensive log of the stuff 
            roll_dict['stats'] = stats
        ppo_trainer.log_stats(stats, batch, logrewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
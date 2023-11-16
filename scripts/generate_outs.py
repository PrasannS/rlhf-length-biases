from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import  AutoTokenizer, AutoModelForCausalLM
from rlhfutils.data import load_rlcd, load_apfarm, webgpt_template
import pandas as pd
import copy
import argparse


def get_step_ckpt(ckpt, origmodel):
    if ckpt=="orig":
        print("using original")
        return copy.deepcopy(origmodel)
    try:   
        return PeftModel.from_pretrained(origmodel, "./trl-stack/"+ckpt)
    except:
        return PeftModel.from_pretrained(origmodel, ckpt)

def load_stack(topval, bottom=0):
    dset = load_dataset("lvwerra/stack-exchange-paired",  data_dir="data/evaluation", split="train")
    dset = dset.shuffle(seed=0)
    dset = dset.select(range(bottom, topval))
    results = []
    for d in range(len(dset)):
        query = "Question: " + dset['question'][d] + "\n\nAnswer: "
        results.append({
            'query':query,
            'response_j':dset['response_j'][d],
            'response_k':dset['response_k'][d],
        })
    return results

def preproc_wgpt(example):
    ex = {}
    ex['question'] = example['question']['full_text']
    if example['score_0']>example['score_1']:
        ex['response_j'] = example['answer_0']
        ex['response_k'] = example['answer_1']
    else:
        ex['response_k'] = example['answer_0']
        ex['response_j'] = example['answer_1']
    return ex

def adjust_input(strval, apf=True):
    if "Input: " in strval:
        strval = strval.replace("Input: ", "### Input:\n")
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+strval+"\n\n### Response:"

def load_webgpt(topval, bottom=0):
    # take eval set using specific seed so it's not polluted by RM initial stuff
    tdset = load_dataset("openai/webgpt_comparisons", split="train")
    tdset = tdset.map(preproc_wgpt)
    tdset = tdset.shuffle(seed=100)
    # take the last portion as a test set
    dset = tdset.select(range(18000, len(tdset)))
    dset = dset.select(range(bottom, topval))
    results = []
    for d in range(len(dset)):
        # use apfarm prompt format here
        query = adjust_input(dset['question'][d])
        results.append({
            'query':query,
            'response_j':dset['response_j'][d],
            'response_k':dset['response_k'][d],
        })
    return results

def adjust_rlcd(question):
    # NOTE this only works for new RLCD models, prompt matters, this was off, rerun wgpt accordingly
    return webgpt_template("Continue the conversation:\n\n"+question.strip()+"\n\nAssistant: ")
# wrapper for loading rlcd
def lrlcd(topval, bottom=0):
    _, eval_dset = load_rlcd()
    
    # NOTE this is very hacky as I'm just slamming things in
    res = [{'query':adjust_rlcd(q['question'])} for q in eval_dset]
    return res[bottom:topval]
    
# wrapper for  loading apf
def lapf(topval, bottom=0):
    _, eval_dset = load_apfarm("gpt4")
    # NOTE need to get the prompt data in the right way, not the RM way
    res = [{'query':adjust_input(q['question'], True)} for q in eval_dset]
    
    print("SANITY CHECK\n"+res[0]['query'])
    return res[bottom:topval]

# TODO maybe clean this up for easier runs later on
def load_dset(dset, topval, bottom=0):
    if 'stack' in dset:
        return load_stack(topval, bottom)
    elif 'webgpt' in dset:
        return load_webgpt(topval, bottom)
    # TODO not yet certain if prompts, etc. are matching up 
    elif "apf" in dset:
        return lapf(topval, bottom)
    elif "rlcd" in dset:
        return lrlcd(topval, bottom)

def generate_outs(model, results, generation_kwargs, qsize=1, savefile="tmp.jsonl"):
    generation_kwargs['num_return_sequences']=1
    scored_results = []
    with torch.no_grad():
        qtemps = []
        curcnt = 0
        for result in tqdm(results, desc='Processing results'):
            qtemps.append(result['query'])
            if (curcnt+1)%qsize==0:
                generated_responses = []
                try: 
                    model_inputs = tokenizer(qtemps, return_tensors='pt', padding=True, truncation=True).to(model.device)
        
                    # Generate outputs for N things in one go
                    generated_output = [model.generate(**model_inputs, **generation_kwargs)]
                except:
                    # if batch is too big then split it up
                    generated_output = []
                    print("Got an OOM error")
                    torch.cuda.empty_cache()
                    for i in range(0, 6, 2):
                        model_inputs = tokenizer(qtemps[i:i+2], return_tensors='pt', padding=True, truncation=True).to(model.device)
                        generated_output.append(model.generate(**model_inputs, **generation_kwargs))
                for gen in generated_output:
                    for generated_sequence in gen:
                        # HACK to see if the huggingface issue was the problem
                        decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
                        generated_responses.append(decoded_sequence)
        
                # Append scored results
                for i in range(len(generated_responses)):
                    # we're only generating for 1 thing at a time
                    scored_results.append({
                        'question': qtemps[i],
                        'response': generated_responses[i],
                        #'score': score
                    })
                pd.DataFrame(scored_results).to_json(savefile, orient='records', lines=True)
                qtemps = []
            curcnt = curcnt+1

    return scored_results

# generate_outs but for cases where we want to generate a distribution 
def multi_generate_outs(model, results, generation_kwargs, bsize=1, savefile="tmp.jsonl"):
    generation_kwargs['num_return_sequences']=bsize
    scored_results = []
    with torch.no_grad():
        curcnt = 0
        for result in tqdm(results, desc='Processing results'):            
            generated_responses = []
            model_inputs = tokenizer([result['query']], return_tensors='pt', padding=True, truncation=True).to(model.device)
            try: 
                # Generate outputs for N things in one go
                generated_output = [model.generate(**model_inputs, **generation_kwargs)]
            except:
                generation_kwargs['num_return_sequences']=2
                # if batch is too big then split it up
                generated_output = []
                print("Got an OOM error")
                torch.cuda.empty_cache()
                for i in range(0, bsize, 2):
                    generated_output.append(model.generate(**model_inputs, **generation_kwargs))
            for gen in generated_output:
                for generated_sequence in gen:
                    # HACK to see if the huggingface issue was the problem
                    decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
                    generated_responses.append(decoded_sequence)
    
            # Append scored results
            
            # we're only generating for 1 thing at a time
            scored_results.append({
                'question': result['query'],
                'response': generated_responses,
                #'score': score
            })
            pd.DataFrame(scored_results).to_json(savefile, orient='records', lines=True)
            curcnt = curcnt+1

    return scored_results    

def main(args):
    # NOTE, make sure to set CUDA_VISIBLE_DEVICES in a call
    # load in original sft model
    origmodel = AutoModelForCausalLM.from_pretrained(
        args.basemodel,
        load_in_8bit=True, # re-enable for llama model
        device_map={"": 0},
    )
    print("original model loaded")

        
    print(tokenizer.decode(tokenizer.eos_token_id))
    print(tokenizer.decode(tokenizer.pad_token_id))
        
    # NOTE try original kwargs since new ones are broken?
    generation_kwargs = {
        "min_length": 200, # TODO switch this back at some point
        "max_new_tokens":256,
        #"top_k": 0.0,
        "top_p": 0.9,
        "temperature": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    results = load_dset(args.dset, args.top, args.bottom)
    print(results[0]['query'])
    # ckpts = ["/mnt/data1/prasann/rlhf-exploration/stack-llama/checkpoints/advmseppo/step_125", "orig", "/mnt/data1/prasann/rlhf-exploration/stack-llama/checkpoints/2gpumix"]
    # fnames = ["advmse", "orig", "mix"]
    ckpts = [args.ckname]
    fnames = [args.fname]
    
    # ckpts = [s.replace("oldrm", "rlhfdalen") for s in ckpts]
    # fnames = [s.replace("olds", "dalenl") for s in fnames]
    # fnames = ["oldrmouts", "daouts", "origouts"]
    
    # Repeat generation process for each relevant checkpoint
    for i in range(len(ckpts)):
        allres = []
        print("going through process for checkpoint "+str(ckpts[i]))
        fname = "generated_"+str(fnames[i])+".jsonl"
        model = get_step_ckpt(ckpts[i], origmodel)
        if args.bsize>1:
            multi_generate_outs(model, results, generation_kwargs, args.bsize, fname)
        else:
            generate_outs(model, results, generation_kwargs, 6, fname)
        
        # TODO is model deletion necessary?
        del model

if __name__=="__main__":
    
    # take in args and parse them
    parser = argparse.ArgumentParser(description='My Python script.')
    parser.add_argument('basemodel', type=str, help='base model checkpoint is trained on')
    parser.add_argument('dset', type=str, help='name of dataset to generate on')
    parser.add_argument('ckname', type=str, help='checkpoint to load from')
    parser.add_argument('fname', type=str, help='generation filename')
    parser.add_argument('bottom', type=int, help='bottom of range to generate for')
    parser.add_argument('top', type=int, help='top of range to generate for')
    parser.add_argument('bsize', type=int, help='outputs per prompt')
    
    progargs = parser.parse_args()
    # make tokenizer, get stuff started
    tokenizer = AutoTokenizer.from_pretrained(progargs.basemodel, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    main(progargs)
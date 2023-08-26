from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import  AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline, AutoTokenizer
import pandas as pd
import copy


model_name = "./models/sft"

def get_step_ckpt(ckpt, origmodel):
    if ckpt=="orig":
        print("using original")
        return copy.deepcopy(origmodel)
    try:   
        return PeftModel.from_pretrained(origmodel, "./trl-stack/"+ckpt)
    except:
        return PeftModel.from_pretrained(origmodel, ckpt)

def load_dset(topval, bottom=0):
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
                    model_inputs = tokenizer(qtemps, return_tensors='pt', return_token_type_ids=False, padding=True, truncation=True).to(model.device)
        
                    # Generate outputs for N things in one go
                    generated_output = [model.generate(**model_inputs, **generation_kwargs)]
                except:
                    # if batch is too big then split it up
                    generated_output = []
                    print("Got an OOM error")
                    torch.cuda.empty_cache()
                    for i in range(0, 6, 2):
                        model_inputs = tokenizer(qtemps[i:i+2], return_tensors='pt', return_token_type_ids=False, padding=True, truncation=True).to(model.device)
                        generated_output.append(model.generate(**model_inputs, **generation_kwargs))
                for gen in generated_output:
                    for generated_sequence in gen:
                        # HACK to see if the huggingface issue was the problem
                        decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=False)
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

if __name__=="__main__":
    # NOTE, make sure to set CUDA_VISIBLE_DEVICES in a call
    # load in original sft model
    origmodel = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True, # re-enable for llama model
        device_map={"": 0},
        #peft_config=lora_config,
    )
    print("original model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #if getattr(tokenizer, "pad_token", None) is None:
    #    tokenizer.pad_token = tokenizer.eos_token
        
    print(tokenizer.decode(tokenizer.eos_token_id))
    
    generation_kwargs = {
        "min_new_tokens": -1,
        "max_new_tokens":256,
        #"top_k": 0.0,
        "top_p": 0.9,
        "temperature": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.2,
        #"pad_token_id": tokenizer.pad_token_id,
        #"eos_token_id": tokenizer.eos_token_id,
    }
    
    results = load_dset(800, 0)

    ckpts = ["/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advrandonlyppo/step_175"]
    fnames = ["randonly175"]
    
    # ckpts = [s.replace("oldrm", "rlhfdalen") for s in ckpts]
    # fnames = [s.replace("olds", "dalenl") for s in fnames]
    #fnames = ["oldrmouts", "daouts", "origouts"]
    
    # Repeat generation process for each relevant checkpoint
    for i in range(len(ckpts)):
        allres = []
        print("going through process for checkpoint "+str(ckpts[i]))
        fname = "generated_"+str(fnames[i])+".jsonl"
        model = get_step_ckpt(ckpts[i], origmodel)
        generate_outs(model, results, generation_kwargs, 6, fname)
        
        # TODO is model deletion necessary?
        del model
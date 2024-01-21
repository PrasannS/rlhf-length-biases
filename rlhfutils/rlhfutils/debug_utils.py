from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import copy
import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import difflib
from IPython.core.display import display, HTML
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

def get_omodel(base):
    origmodel = AutoModelForCausalLM.from_pretrained(
        base,
        load_in_8bit=True, # re-enable for llama model
        device_map={"": 0},
        #peft_config=lora_config,
    )
    return origmodel, AutoTokenizer.from_pretrained(base)

def adjust_input_apf(strval, response=None):
    qstr = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+strval+"\n\n### Response:"
    if response:
        return qstr+response
    else:
        return qstr

def adjust_input_stack(strval, response=None):
    qstr = "Question: " + strval + "\n\nAnswer: "
    if response:
        return qstr+response
    else:
        return qstr

def get_peft_ckpt(base, fname):
    if fname=="orig":
        print("using original")
        return copy.deepcopy(base)
    return PeftModel.from_pretrained(base, fname)

def generate_outputs(inpstrs, toker, mod, kwargs):
    inps = toker(inpstrs, return_tensors="pt").to(mod.device)
    outs = mod.generate(**inps, **kwargs)
    return outs , toker.batch_decode(outs, skip_special_tokens=False)

def load_all_dfs(dir):
    res = {}
    for f in os.listdir(dir):
        if ".jsonl" in f:
            name = f.replace("generated_", "").replace(".jsonl", "")
            res[name] = pd.read_json(dir+f, orient='records', lines=True)
            res[name]['answer'] = [s.split("Answer:")[1] for s in res[name]['response']]
    return res

def load_all_rmdfs(dir):
    res = {}
    for f in os.listdir(dir):
        if ".jsonl" in f:
            name = f.replace(".jsonl", "")
            tmp = pd.read_json(dir+f, orient='records', lines=True)
            if "shuff" in name:
                tmp['reward'] = tmp[name.replace("shuff", "")]
            else:
                tmp['reward'] = tmp[name]
            if "Below is an instruction" in tmp['question'][0]:
                res["wgpt_"+name] = tmp
            else:
                res['stack_'+name] = tmp
    return res

def load_all_hackdfs(base):
    alldfs = {}
    for f in os.listdir(base):
        if ".jsonl" in f:
            tmp = pd.read_json(base+f, lines=True, orient='records')
            tmp = tmp.dropna()
            tmp['maxsco'] = [max(m.bestscos+[m.origsco]) for i, m in tmp.iterrows()]
            tmp['diff'] = tmp['maxsco'] - tmp['origsco']
            alldfs[f.replace(".jsonl", "")] = tmp
        
    return alldfs
    
def load_rm(name, device, quant=True, basemodel="nobase", doeval=False, tokenwise=False):
    tokdir = name if "nobase" in basemodel else basemodel

    tokenizer = AutoTokenizer.from_pretrained(tokdir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 8,
        "truncation": True,
    }
    # we want to load model in directly as an adapter
    if "nobase" not in basemodel:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(
                basemodel, num_labels=1, torch_dtype=torch.bfloat16
            )
            # load in stuff based on adapter
            model = PeftModel.from_pretrained(model, name)
            if doeval:
                model.eval()
                model.to(device)
                
                return tokenizer, model, None
            # NOTE pipeline not supported by peft
            return tokenizer, model, None
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                basemodel, num_labels=1, torch_dtype=torch.bfloat16
            )
            # load in stuff based on adapter
            model = PeftModel.from_pretrained(model, name)
            if doeval:
                model.eval()
                model.to(device)
                sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    return_token_type_ids=False,
                )
                return tokenizer, sentiment_pipe, kwargs
            # NOTE pipeline not supported by peft
            return tokenizer, model, None
    else:
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=name,
            device_map={"": device},
            model_kwargs={"load_in_8bit": quant},
            tokenizer=tokenizer,
            return_token_type_ids=False,
        )
    
    return tokenizer, sentiment_pipe, kwargs

def progress_rm(inputs, rm, kwargs, split=8, catcherrs=False):
    results = []
    for i in tqdm(range(0, len(inputs), split)):
        if catcherrs:
            try:
                results.extend(rm(inputs[i:i+split], **kwargs))
            except:
                results.extend([[{'score':None}]]*split)
                torch.cuda.empty_cache()
        else:
            results.extend(rm(inputs[i:i+split], **kwargs))
    return results

def highlight_differences(old, new):
    d = difflib.Differ()
    diff = list(d.compare(old, new))
    
    display_str = ""
    
    for s in diff:
        if s[0] == ' ':
            display_str += s[2:]
        elif s[0] == '-':
            display_str += f"<span style='background-color: red'>{s[2:]}</span>"
        elif s[0] == '+':
            display_str += f"<span style='background-color: green'>{s[2:]}</span>"
    
    display(HTML(display_str))
    
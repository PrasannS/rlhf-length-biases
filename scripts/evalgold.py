from rlhfutils.rewards import get_synth_rewards
import rlhfutils.rewards as rutils
import pandas as pd
from statistics import mean
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
from rlhfutils.eval_utils import scofile

toker = AutoTokenizer.from_pretrained("facebook/opt-125m")

def tokenproc(inp, lim=True, function=None):
    #print(inp)
    #print(function)
    if (function is None) or "contpos" not in function:
        if "\n\nAnswer:" in inp:
            inp = inp.split("\n\nAnswer:")[1]
        elif "### Response:" in inp:
            inp = inp.split("### Response:")[1]
        elif "<assistant>" in inp: 
            inp = inp.split("<assistant>")[1]
        inp = inp.strip()
        start=0
    else:
        # TODO this looks weird
        start = len(toker(inp.split("### Response:")[0]).input_ids)
    if lim:
        tokd = toker(inp).input_ids[:start+50]
    else: 
        tokd = toker(inp).input_ids
    return toker.decode(tokd, skip_special_tokens=True)
        
if __name__=="__main__": 
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Process two string arguments.')

    # Add arguments
    parser.add_argument('--fname', type=str, required=True, help='The filename as a string')
    parser.add_argument('--gfunct', type=str, required=True, help='The function name as a string')
    
    # Execute the parse_args() method
    args = parser.parse_args()
    
    # NOTE special case for distillation reward
    if "contrastivedistill" in args.gfunct:
        rutils.likemod = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        rutils.liketok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        rutils.slikemod = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        rutils.sliketok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    t, lens = scofile(args.fname, args.gfunct, True, 0)
    fv, lens = scofile(args.fname, args.gfunct, False, 0)
    tval, tmeans = t
    fval, fmeans = fv
    
    # use everything except the filename
    outf = args.fname.replace(".jsonl", "")+".results"
    with open(outf, 'w') as f:
        json.dump({
            'truncval':tval,
            'notruncval':fval,
            'meanlen':mean(lens),
            'truncdist':tmeans,
            'notrunctdist':fmeans,
            'lendist':lens
        }, f)
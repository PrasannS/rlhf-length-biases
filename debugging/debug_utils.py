from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import copy

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

def get_peft_ckpt(base, fname):
    if fname=="orig":
        print("using original")
        return copy.deepcopy(base)
    return PeftModel.from_pretrained(base, fname)

def generate_outputs(inpstrs, toker, mod, kwargs):
    inps = toker(inpstrs, return_tensors="pt").to(mod.device)
    outs = mod.generate(**inps, **kwargs)
    return outs , toker.batch_decode(outs, skip_special_tokens=False)
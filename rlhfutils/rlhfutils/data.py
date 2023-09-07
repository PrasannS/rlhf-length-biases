# Code for loading in data for reward modeling and PPO

from datasets import load_dataset
import pandas as pd

# NOTE process anthro hh data for mixture with se data
def preproc_hh(example):
    j = example['chosen']
    hind = j.index("Human:")+6
    aind = j.index("Assistant:")+len("Assistant:")
    ex = {}
    ex['question'] = j[hind:aind-len("Assistant:")]
    ex['response_j'] = example['chosen'][aind:]
    ex['response_k'] = example['rejected'][aind:]
    return ex

def preproc_apf(example):
    ex = {}
    if len(example['input'])>0:
        ex['question'] = example['instruction']+"\n\nInput: "+example['input']
    else:
        ex['question'] = example['instruction']
        
    if example['preference']==2:
        ex['response_j'] = example['output_2']
        ex['response_k'] = example['output_1']
    else:
        ex['response_k'] = example['output_2']
        ex['response_j'] = example['output_1']
    return ex

def preproc_rlcd_all (inpdf):
    allres = []
    for row in inpdf:
        res = {}
        res['question'] = row['instruction']
        if row['preference']==1:
            res['response_j'] = row['output_1']
            res['response_k'] = row['output_2']
        else:
            res['response_j'] = row['output_2']
            res['response_k'] = row['output_1']
        allres.append(res)
    return pd.DataFrame(allres).dropna().reset_index(drop=True)

# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function_rm(examples, tokenizer):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

def tokenize_dset(train_dataset, eval_dataset, script_args, tokenizer):
    
    num_proc = 24  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        lambda example: preprocess_function_rm(example, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
    )

    eval_dataset = eval_dataset.map(
        lambda example: preprocess_function_rm(example, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
    )
    return train_dataset, eval_dataset

# NOTE process anthro hh data for mixture with se data
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

# load in standard dataset for webgpt
def load_wgpt():
    train_dataset = load_dataset("openai/webgpt_comparisons", split="train")
    train_dataset = train_dataset.map(preproc_wgpt)

    print("initial size ", len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=100)
    # take the last portion as a test set
    eval_dataset = train_dataset.select(range(18000, len(train_dataset)))
    # use the rest as necessary
    train_dataset = train_dataset.select(range(18000))

    print("new size ", len(train_dataset))
    return train_dataset, eval_dataset

# TODO maybe same as HH, can compress those 2?
def preproc_rlcd(example):
    ex = {}
    ex['question'] = example['instruction'][len("Human: "):-len("\n\nAssistant:")]
    if example['preference']==1:
        ex['response_j'] = example['output_1']
        ex['response_k'] = example['output_2']
    else:
        ex['response_k'] = example['output_1']
        ex['response_j'] = example['output_2']
    return ex

# HACK assume all RM training is happening on Fuji
BASE="/home/prasann/Projects/rlhf-exploration/"
def load_rlcd():
    train_dataset = load_dataset("csv", data_files=BASE+"rlcd-llama/simulated_data/simulated_preference_data_consolidated_helpful7b.csv")['train']
    
    # HACK against 2 weird NaN outputs
    train_dataset = train_dataset.filter(
        lambda x: x['output_1'] and x['output_2']
    )
    train_dataset = train_dataset.map(preproc_rlcd)

    print("initial size ", len(train_dataset))

    train_dataset = train_dataset.shuffle(seed=100)
    # take the last portion as a test set, HACK this value is hard-coded 
    eval_dataset = train_dataset.select(range(40000, len(train_dataset)))
    # use the rest as necessary
    train_dataset = train_dataset.select(range(40000))
    
    return train_dataset, eval_dataset

def load_stack():
    train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
    print("initial size ", len(train_dataset))

    # HACK hardcoded and there's some inconsistency here to be careful of
    train_dataset = train_dataset.select(range(100000))
    train_dataset = train_dataset.shuffle(seed=0)

    print("new size ", len(train_dataset))

    eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
    eval_dataset = eval_dataset.select(range(int(len(train_dataset)*0.01)))
    
    eval_dataset = eval_dataset.shuffle(seed=0) # redundant? 
    
    return train_dataset, eval_dataset

trainperc =  0.95
def load_apfarm(dname):
    # 2 different datasets within 1
    if "gpt" in dname:
        print("loading gpt4 dataset")
        train_dataset = load_dataset("tatsu-lab/alpaca_farm", 'alpaca_gpt4_preference')['preference']
    else:
        print("loading human dataset")
        train_dataset = load_dataset("tatsu-lab/alpaca_farm", 'alpaca_human_preference')['preference']
        
    train_dataset = train_dataset.map(preproc_apf)

    print("initial size ", len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=100)
    # use the rest as necessary
    train_dataset = train_dataset.select(range(int(trainperc*len(train_dataset))))

    print("new size ", len(train_dataset))
    # take the last portion as a test set
    eval_dataset = train_dataset.select(range(int(trainperc*len(train_dataset)), len(train_dataset)))
    
    return train_dataset, eval_dataset

#################### Code for prompt templates / RL stuff  #######################
# TODO may merge with apf, or maybe go into data

def mapfilt(ds, tkfunct):
    ds = ds.map(tkfunct, batched=True)
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds

def webgpt_template(strval, resp=None):
    if resp:
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+strval+"\n\n### Response:"+resp
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+strval+"\n\n### Response:"

# Build WGPT Dataset for RL (may be able to compress)
def build_wgpt_promptdata(tokenizer):

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)
    ds = load_dataset("openai/webgpt_comparisons", split="train")

    def tokwgpt(sample):
        # TODO trying out this thing for batching
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in sample["question"]:
            query = question['full_text']
            
            query = webgpt_template(query)
            
            #query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(question['full_text'])
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    return mapfilt(ds, tokwgpt)

def build_rlcd_promptdata(tokenizer):

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")

    def tokrlcd(sample):
        # TODO trying out this thing for batching
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in sample["chosen"]:
            
            # NOTE that HHRLHF, rlcd are multi-turn, which means there's turn context (setting is a bit different as a result)
            hind = question.index("Human:")+6
            aind = question.index("Assistant:")+len("Assistant:")
            qstr = question[hind:aind-len("Assistant:")]
            
            query = webgpt_template(qstr)
            
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    return mapfilt(ds, tokrlcd)


def build_stack_promptdata(tokenizer):
    # load stack with datasets
    ds = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
    
    def tokstack(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    return mapfilt(ds, tokstack)

def adjust_apf(instr, inp):
    if len(inp)==0:   
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+instr+"\n\n### Response:"
    # version that also has the input
    return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n"+instr+"\n\n### Input:\n"+inp+"\n\n### Response:"

def inp_origformat(instr, inp):
    if len(inp)>0:
        return "Question: "+instr+"\n\nInput: "+inp+"\n\nAnswer: "
    else:
        return "Question: "+instr+"\n\nAnswer: "

def build_apf_promptdata(tokenizer):

    ds = load_dataset("tatsu-lab/alpaca_farm", 'alpaca_instructions')['unlabeled']
 
    def apftok(sample):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for i in range(len(sample['instruction'])):
            # get apfarm processed input
            query = adjust_apf(sample['instruction'][i], sample['input'][i])
            
            #query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(inp_origformat(sample['instruction'][i], sample['input'][i]))
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    return mapfilt(ds, apftok)

# Format used for QA style reward models (WebGPT, AlpacaFarm?, Stack)
def qaform(q, r):
    return "Question: " + q + "\n\nAnswer: " + r

def anscat(q, r):
    return q+r

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
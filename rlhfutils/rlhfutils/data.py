# Code for loading in data for reward modeling and PPO

from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
import pandas as pd
import random
from typing import List, Callable
from dataclasses import dataclass

# NOTE process anthro hh data for mixture with se data
def preproc_hh(example):
    j = example['chosen']
    hind = j.index("Human:")+6
    aind = j.rindex("Assistant:")+len("Assistant:")
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

def preproc_rlcd_all(inpdf):
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
        "ids": [], # used for data carto
    }
    hasmag = False
    if "magnitude" in examples.keys():
        new_examples['mag'] = [] # handle ultra case
        hasmag = True
    else:
        examples['magnitude'] = [None]*len(examples["question"])
        
        
    for question, response_j, response_k, rowid, mag in zip(examples["question"], examples["response_j"], examples["response_k"], examples['row_index'], examples['magnitude']):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
        # get hash-based id for data carto, this will allow for selection of examples later on
        # TODO do without hashing
        # new_examples['ids'].append(str(hash(response_j))+"_"+str(hash(response_k)))
        new_examples['ids'].append(rowid)
        if hasmag:
            new_examples['mag'].append(mag)

    return new_examples

def mix_hh(train_dset, ratio):
    hh_train = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    hh_train = hh_train.map(preproc_hh)
    hh_train = hh_train.shuffle(seed=100)
    hh_train = hh_train.select(range(int(len(train_dset)*ratio)))
    return hh_train

def augment_random(example, dset):
    # Get a random index
    index = random.randint(0, len(dset)-1)

    # Get the sample at the random index
    example['response_k'] = dset[index]['response_j']
    return example

def modify_dataset(functions: List[Callable], dataset: Dataset, props: List[float]) -> Dataset:
    #assert sum(counts) <= len(dataset), "Counts sum should not exceed dataset size"
    assert len(functions) == len(props), "Number of functions should equal number of counts"

    # convert ratios into actual counts of data
    counts = [int(p*len(dataset)) for p in props]
    # Shuffle dataset
    # dataset = dataset.shuffle(seed=0)

    # Keep track of modified parts of the dataset
    modified_datasets = []

    start_index = 0
    for fn, count in zip(functions, counts):
        print(fn.__name__)
        end_index = start_index + count

        # If function takes in both the example and the dataset
        if fn.__code__.co_argcount == 2:
            modified_datasets.append(dataset.select(list(range(start_index, end_index))).map(lambda example: fn(example, dataset)))

        # If function only takes in the example
        elif fn.__code__.co_argcount == 1:
            modified_datasets.append(dataset.select(list(range(start_index, end_index))).map(fn))

        start_index = end_index

    # Add the unmodified part of the dataset
    if start_index < len(dataset):
        modified_datasets.append(dataset.select(list(range(start_index, len(dataset)))))

    # Concatenate all parts of the dataset back together
    return concatenate_datasets(modified_datasets)

@dataclass
class tmpdata: 
    mix_ratio: str
    rand_ratio:  str
    
def augment_data(train_dset, script_args, justaugdata=False):
    augdata = []
    # we can sub in initial dataset for 2 types of DA 
    if script_args.rand_ratio > 0:
        if justaugdata:
            augdata.append(modify_dataset([augment_random], train_dset.select(range(int(len(train_dset)*script_args.rand_ratio))), [1.0]))
        else:
            print("mixing in random data")
            re_add = train_dset.select(range(int(len(train_dset)*script_args.rand_ratio)))
            train_dset = modify_dataset([augment_random], train_dset, [script_args.rand_ratio])
                
            # add back in data that was randomized
            train_dset = concatenate_datasets([train_dset, re_add])
            train_dset = train_dset.to_pandas()
            train_dset['isrand'] = 0
            train_dset['isrand'].iloc[:int(len(re_add))] = 1
            train_dset = Dataset.from_pandas(train_dset)
    if script_args.mix_ratio > 0:
        if justaugdata:
            augdata.append(mix_hh(train_dset, script_args.mix_ratio))
        else:
            print("mixing in HH data")
            mixdset = mix_hh(train_dset, script_args.mix_ratio)
            train_dset = concatenate_datasets([train_dset, mixdset])
            train_dset = train_dset.to_pandas()
            train_dset['ismix'] = 0
            train_dset['ismix'].iloc[:-int(len(mixdset))] = 1
            train_dset = Dataset.from_pandas(train_dset)
    if justaugdata:
        if len(augdata)==0:
            return None
        # HACK can do concat / shuffle later if necessary
        return concatenate_datasets(augdata)
    return train_dset.shuffle(seed=100)

def baldf(indf):
    df = indf.copy()
    # Create bins of 10
    bins = range(-200, 201, 10)
    df['bin'] = pd.cut(df['diffv'], bins=bins)
    
    # Initialize an empty DataFrame to store balanced data
    balanced_df = pd.DataFrame()
    
    # Get unique bin labels from the DataFrame
    unique_bins = df['bin'].dropna().unique()
    
    # Iterate through each pair of negative and positive bins
    for bin_label in unique_bins:
        if bin_label.left >= 0:
            continue
    
        # Find the positive counterpart of the current negative bin
        positive_bin = pd.Interval(-bin_label.right, -bin_label.left)
    
        # If the positive counterpart is not in unique_bins, skip this iteration
        if positive_bin not in unique_bins:
            continue
    
        # Find the counts for the negative and positive bins
        neg_count = df[df['bin'] == bin_label].shape[0]
        pos_count = df[df['bin'] == positive_bin].shape[0]
    
        # Find the minimum count to balance the data
        min_count = min(neg_count, pos_count)
    
        # Randomly sample min_count rows from each bin and append to balanced_df
        sampled_neg = df[df['bin'] == bin_label].sample(min_count, random_state=0)
        sampled_pos = df[df['bin'] == positive_bin].sample(min_count, random_state=0)
    
        balanced_df = pd.concat([balanced_df, sampled_neg, sampled_pos])
    
    # Reset index of the balanced DataFrame
    balanced_df.reset_index(drop=True, inplace=True)
    balanced_df = balanced_df.drop(columns=['bin'])
    # Now balanced_df contains the balanced data
    return balanced_df
    
def len_balance(indataset):
    # Convert the Hugging Face dataset to a pandas DataFrame
    dataset_df = pd.DataFrame(indataset)
    lj = [len(ts) for ts in dataset_df['input_ids_j']]
    lk = [len(ts) for ts in dataset_df['input_ids_k']]
    dataset_df['diffv'] =[lj[i] - lk[i] for i in range(len(lj))]
    balanced_df = baldf(dataset_df)
    
    print(balanced_df)
    # # Create a new column to identify which row has longer input_ids_j
    # dataset_df['j_is_longer'] = dataset_df.apply(lambda row: len(row['input_ids_j']) > len(row['input_ids_k']), axis=1)

    # # Count how many examples are in each class
    # count_j_is_longer = len(dataset_df[dataset_df['j_is_longer'] == True])
    # count_k_is_longer = len(dataset_df) - count_j_is_longer

    # # Determine which class has fewer examples
    # min_count = min(count_j_is_longer, count_k_is_longer)

    # # Downsample the class with more examples to have as many as the class with fewer examples
    # if count_j_is_longer > min_count:
    #     df_j_is_longer = dataset_df[dataset_df['j_is_longer'] == True].sample(n=min_count)
    #     df_k_is_longer = dataset_df[dataset_df['j_is_longer'] == False]
    # else:
    #     df_k_is_longer = dataset_df[dataset_df['j_is_longer'] == False].sample(n=min_count)
    #     df_j_is_longer = dataset_df[dataset_df['j_is_longer'] == True]

    # # Concatenate the two downsampled classes to create a balanced DataFrame
    # balanced_df = pd.concat([df_j_is_longer, df_k_is_longer])

    # # Optionally, shuffle the rows
    # balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    return  Dataset.from_pandas(balanced_df)

def tokenize_dset(train_dataset, eval_dataset, script_args=None, tokenizer=None):
    selcols = ['question', 'response_j', 'response_k', 'row_index']
    # if we have a magnitude based dataset, make sure to use that automatically 
    if 'magnitude' in train_dataset.column_names:
        # TODO can set smth up to test whether it's necessary or not
        selcols =  selcols+["magnitude"]
    train_dataset = train_dataset.select_columns(selcols)
    num_proc = 10  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names
    
    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        lambda example: preprocess_function_rm(example, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    if script_args is not None:
        mlen = script_args.max_length
    else:
        mlen = 512
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= mlen and len(x["input_ids_k"]) <= mlen
    )

    eval_dataset = eval_dataset.map(
        lambda example: preprocess_function_rm(example, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= mlen and len(x["input_ids_k"]) <= mlen
    )

    if script_args is not None and script_args.balance_len==1:
        print("Balancing")
        train_dataset = len_balance(train_dataset)
    
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
def load_rlcd():
    try:
        train_dataset = load_dataset("csv", data_files="/scratch/cluster/prasanns/research/rlhf-exploration/rlcd-llama/simulated_data/simulated_preference_data_consolidated_helpful7b.csv")['train']
    except:
        try:
            train_dataset = load_dataset("csv", data_files="/scratch/cluster/prasanns/research/rlhf-exploration/rlcd-llama/simulated_data/simulated_preference_data_consolidated_helpful7b.csv")['train']
        except:
            train_dataset = load_dataset("csv", data_files="/scratch/cluster/prasanns/research/rlhf-exploration/rlcd-llama/simulated_data/simulated_preference_data_consolidated_helpful7b.csv")['train']
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

def load_harmless():
    train_dataset = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')['train']
    print("initial size ", len(train_dataset))

    train_dataset = train_dataset.shuffle(seed=0)

    eval_dataset = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')['test']
    
    # run the right preprocessing function to prep the data in the right way
    train_dataset = train_dataset.map(preproc_hh)
    eval_dataset = eval_dataset.map(preproc_hh)
    
    return train_dataset, eval_dataset

def load_stack():
    orig_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
    print("initial size ", len(orig_dataset))

    # HACK hardcoded and there's some inconsistency here to be careful of
    train_dataset = orig_dataset.select(range(100000))
    train_dataset = train_dataset.shuffle(seed=0)
    # add in 150K more examples for dcarto
    # train_dataset = concatenate_datasets([train_dataset, orig_dataset.select(range(100000,250000))])
    print("new size ", len(train_dataset))

    eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
    eval_dataset = eval_dataset.select(range(int(len(train_dataset)*0.01)))
    
    eval_dataset = eval_dataset.shuffle(seed=0) # redundant? 
    
    return train_dataset, eval_dataset

def load_manual(iname, base="data/categories/"):
   
    print("GOING THROUGH PROCESS FOR "+iname)
    if "stack" in iname:
        orig_dataset = load_from_disk(base+iname.replace("stack_", ""))
    else: 
        orig_dataset = load_from_disk(base+iname)
    print("initial size ", len(orig_dataset))

    # NOTE use 95% of the dataset for training
    DRATIO = 0.99
    train_dataset = orig_dataset.select(range(int(len(orig_dataset)*DRATIO)))
    print(len(train_dataset))
    eval_dataset = orig_dataset.select(range(int(len(orig_dataset)*DRATIO), len(orig_dataset)))
    print("eval len")
    print(len(eval_dataset))
    return train_dataset, eval_dataset
    
    
# load data for ultra-feedback dataset
def load_ultra(dname="data/ultrafeeddiff", useall=False):
    # need to first generate data and store it in the right directory
    # NOTE can also try doing smth with the data that's supposed to be "equal"
    orig_dataset = load_from_disk(dname)
    print("initial size ", len(orig_dataset))
    
    orig_dataset = orig_dataset.shuffle(seed=0)
    if useall:
        return orig_dataset, orig_dataset
    # NOTE use 95% of the dataset for training
    DRATIO = 0.99
    if len(orig_dataset)<30000:
        DRATIO = 0.9
    train_dataset = orig_dataset.select(range(int(len(orig_dataset)*DRATIO)))
    print(len(train_dataset))
    eval_dataset = orig_dataset.select(range(int(len(orig_dataset)*DRATIO), len(orig_dataset)))
    print("eval len")
    print(len(eval_dataset))
    return train_dataset, eval_dataset

# load data for ultra-feedback dataset
def load_stack_mag():
    # need to first generate data and store it in the right directory
    # NOTE can also try doing smth with the data that's supposed to be "equal"
    orig_dataset = load_from_disk("../data/ultrafeeddiff")
    print("initial size ", len(orig_dataset))
    
    orig_dataset = orig_dataset.shuffle(seed=0)
    # NOTE use 95% of the dataset for training
    DRATIO = 0.95
    train_dataset = orig_dataset.select(range(int(len(orig_dataset)*DRATIO)))

    eval_dataset = eval_dataset.select(range(int(len(orig_dataset)*DRATIO)), len(orig_dataset))
        
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

# Build WGPT Dataset for RL (may be able to compress)
def build_ultra_promptdata(tokenizer):
    # input_size = LengthSampler(input_min_text_length, input_max_text_length)
    ds = load_dataset("stingning/ultrachat", split="train")

    def tokultra(sample):
        # TODO trying out this thing for batching
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in sample["data"]:
            # just use the first thing as the prompt every time? 
            query = question[0]
            
            query = webgpt_template(query)
            
            #query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(question[0])
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    return mapfilt(ds, tokultra)

def build_rlcd_promptdata(tokenizer, dname):

    if "harm" in dname:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    else:
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
            # NOTE this maybe makes things weird? TODO should I rerun things without doing this
            aind = question.rindex("Assistant:")+len("Assistant:")
            qstr = question[hind:aind-len("Assistant:")]
            
            # NOTE this only works for new RLCD models, prompt matters, this was off, rerun wgpt accordingly
            query = webgpt_template("Continue the conversation:\n\n"+qstr.strip()+"\n\nAssistant: ")
            
            tokenized_question = tokenizer(query, truncation=True)
            # HACK RM gets the webgpt format
            new_examples["query"].append("Question: " + qstr.strip() + "\n\nAnswer: ")
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    return mapfilt(ds, tokrlcd)


def build_stack_promptdata(tokenizer):
    # load stack with datasets
    ds = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
    ds = ds.select(range(0, 500000))
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
    if inp==None or len(inp)==0:   
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+instr+"\n\n### Response:"
    # version that also has the input
    return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n"+instr+"\n\n### Input:\n"+inp+"\n\n### Response:"

def inp_origformat(instr, inp):
    if inp!=None and len(inp)>0:
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
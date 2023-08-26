import os
import openai
from tqdm import tqdm
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
import pandas as pd
from statistics import mean, stdev
from transformers import AutoTokenizer

toker = AutoTokenizer.from_pretrained("../stack-llama/models/sft")
toker.pad_token_id = toker.eos_token_id

# get openai key from file
def oai_kwargs():
    with open("../secret/openaikey.txt", 'r') as f:
        keyval = f.read()
    if 'OPENAI_API_KEY' not in os.environ:
        decoding_kwargs = dict(
            openai_api_key = keyval, #"sk-...",
            openai_organization_ids = None, # ["org-...","org-..."] you can set multiple orgs to avoid rate limits
        )
        assert decoding_kwargs["openai_api_key"] is not None, "OPENAI_API_KEY not found you should set it in environment or above"
    else:
        decoding_kwargs = {}
    return decoding_kwargs

def count_words(indf, k):
    wcnt = [len(s.split(" ")) for s in indf[k]]
    return wcnt


def reconvert(inp):
    strnew = inp.replace("</s>", "")
    strnew = strnew.replace("<s>", "")
    strnew = strnew.replace("<unk>", "")
    return strnew
    
def load_alldfs(base="use_outs/"):
    alldfs = {}
    for f in os.listdir(base):
        if ".jsonl" in f:
            tmp = pd.read_json(base+f, lines=True, orient='records')
            if "<s>" in tmp['response'][0]:
                tmp['response'] = [reconvert(r) for r in tmp['response']]
                tmp['question'] = [reconvert(r) for r in tmp['question']]
            if "Question:" in tmp['response'][0]:
                tmp['response'] = [r[len(q):] for r, q in zip(tmp['response'], tmp['question'])]
            if "Question:" in tmp['question'][0]:
                tmp['question'] = [s[len("Question: "):-1*len("\n\nAnswer: ")] for s in tmp['question']]
            # constrain to only 200 examples for everything
            #tmp = tmp.loc[:199]
            tmp['wcnt'] = count_words(tmp, "question")
            tmp['rcnt'] = count_words(tmp, "response")
            #tmp = tmp[tmp['wcnts']<200].reset_index()
            #tmp = tmp[tmp['rcnts']<200].reset_index()
            alldfs[f.replace("generated_", "").replace(".jsonl", "")] = tmp
    
    valid_indices_sets = []

    for df_key in alldfs:
        df = alldfs[df_key]
        valid_indices_for_df = set(df[(df['wcnt'] < 1000) & (df['rcnt'] < 1000)].index)
        valid_indices_sets.append(valid_indices_for_df)
    
    # Step 2: Find intersection of all sets to get indices valid across all dataframes
    common_valid_indices = set.intersection(*valid_indices_sets)
    
    # Step 3: Filter each dataframe using the common valid indices
    for df_key in alldfs:
        alldfs[df_key] = alldfs[df_key].loc[list(common_valid_indices)].reset_index(drop=True)
        alldfs[df_key] = alldfs[df_key].sort_values(by='question').reset_index(drop=True)
        alldfs[df_key] = alldfs[df_key].loc[:99]
    #alldfs['davinci']=pd.read_json("../outputs/ckpt_generations/generated_davinci.jsonother", lines=True, orient='records')
    #alldfs['davinci'] = alldfs['davinci'].sort_values(by='question').reset_index(drop=True)
    return alldfs

def apf_format(indf):
    result = []
    for i, row in indf.iterrows():
        result.append({
            'instruction':row['question'],
            'input':"",
            'output':row['response']
        })
    return result

def annotate_apfarm(alldfs, baseline, test, start, end, dec_kwargs):
    # make a fresh annotator each time (is this a good idea?, is it consistent?)
    ann = PairwiseAutoAnnotator(annotators_config="annotators/annotator_pool_v0/configs.yaml", **dec_kwargs)
    base_outs = apf_format(alldfs[baseline])
    test_outs = apf_format(alldfs[test])
    assert end>0
    assert end>start
    base_outs = base_outs[start:end]
    test_outs = test_outs[start:end]
    assert len(base_outs)==len(test_outs)
    annotated = ann.annotate_head2head(outputs_1=base_outs, outputs_2=test_outs)
    dftmp = pd.DataFrame(annotated)
    dftmp.to_json("../outputs/apeval/"+baseline+"_"+test+".jsonl", orient='records', lines=True)
    #pd.DataFrame(annotated).to_json("../outputs/apeval/"+baseline+"_"+test+".jsonl")
    return annotated

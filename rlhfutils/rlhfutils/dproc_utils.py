from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from statistics import mean
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

# process dataframe to give appropriate length information that can be used to get basic length stats
asp_dict = {
    'hf': 'helpfulness',
    'hn': 'honesty', 
    'tn': 'truthfulness', 
    'ifg':'instruction_following'
}

# method to store different kinds of aspect scores, as well as reponse lengths
def process_df_ultra(indf, toker):
    lbl_dict = {'hf': [], 'hn': [], 'tn': [], 'ifg':[]}
    compl_strs = []
    tokens = []
    
    asp_keys = list(asp_dict.keys())

    def process_row(row):
        ltmps = []
        stmps = []

        # TODO add logic to handle mn stuff? 
        lbl_tmp = {k: [] for k in asp_keys}
        cps = row['completions']
        for cp in cps:
            ltmps.append(len(toker(cp['response']).input_ids))
            stmps.append(cp['response'])
            annots = cp['annotations']
            for a in asp_keys:
                lbl_tmp[a].append(annots[asp_dict[a]]['Rating'])
        tokens.append(ltmps)
        compl_strs.append(stmps)
        for a in asp_keys:
            lbl_dict[a].append(lbl_tmp[a])
        return ltmps

    indf['tokens'] = indf.progress_apply(process_row, axis=1)
    indf['resps'] = compl_strs
    for lk in lbl_dict.keys():
        indf[lk] = lbl_dict[lk]

    return indf

def rowmean(row, ind):
    res = []
    for h in asp_dict.keys():
        try:
            res.append(int(row[h][ind]))
        except:
            ""
    return mean(res)

# given dataframe, process values such that mean of values is stored
def procmean(df):
    mnvals = []
    for i, r in df.iterrows():
        try:
            mnvals.append([rowmean(r, i) for i in range(4)])
        except: 
            print("off")
            mnvals.append(-1)
    df['mn'] = mnvals

# convert dataframe into pairwise dataframe for later usage
def create_pairwise_dataframe(df):
    # List to store the new rows for the pairwise dataframe
    new_rows = []

    for _, row in df.iterrows():
        for idx1, idx2 in combinations(range(4), 2):
            idj = idx1
            idk = idx2
            # Determine response_j and response_k based on scores
            if row['mn'][idx1] > row['mn'][idx2]:
                response_j = row['resps'][idx1]
                response_k = row['resps'][idx2]
                magnitude = row['mn'][idx1] - row['mn'][idx2]
                
            elif row['mn'][idx1] < row['mn'][idx2]:
                response_j = row['resps'][idx2]
                response_k = row['resps'][idx1]
                magnitude = row['mn'][idx2] - row['mn'][idx1]
                idj = idx2
                idk = idx1
            else:  # Randomly choose response_j and response_k if scores are equal
                if np.random.choice([True, False]):
                    response_j = row['resps'][idx1]
                    response_k = row['resps'][idx2]
                else:
                    response_j = row['resps'][idx2]
                    response_k = row['resps'][idx1]
                magnitude = 0

            new_rows.append({
                # construct rows according to source, model, etc. This gives some extra surfaces to analyze
                'question': row['instruction'],
                'source':row['source'],
                'modj':row['models'][idj],
                'modk':row['models'][idk],
                'tokj':row['tokens'][idj],
                'tok':row['tokens'][idk],
                'response_j': response_j,
                'response_k': response_k,
                'magnitude': magnitude
            })
    return pd.DataFrame(new_rows)
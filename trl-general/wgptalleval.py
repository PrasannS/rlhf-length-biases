from rlhfutils.eval_utils import oai_kwargs, load_alldfs, annotate_apfarm, apf_format, load_wgpt, filter_and_sort_df
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import re
from transformers import AutoTokenizer
from datasets import load_dataset
import openai
from rlhfutils.data import qaform

toker = AutoTokenizer.from_pretrained("../webgpt-llama/models/sft10k")
adfs = load_alldfs("../trl-general/genouts/lastevalsstack/",  400)

print(adfs.keys())

# trykeys = [ 'stackrwscale', 'stackbalance2', 'stackrda2']
# trykeys = [ 'stackhkl', 'stackbothcut']
trykeys = [ 'stacklenonlyppo3']
#trykeys = ['rlcdlenonly', 'rlcdbalancerm', 'rlcdbothcut', 'rlcdhkl', 'rlcdlenpen']
assert len(adfs[trykeys[0]])>350

for t in trykeys:
    assert t in adfs.keys()

for k in trykeys:
    print(len(adfs[k]))
    # match everything against original PPO
    lenannot = annotate_apfarm(adfs, k, "stackorigrerun", 100, len(adfs[k]), oai_kwargs())
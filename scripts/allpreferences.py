from rlhfutils.eval_utils import oai_kwargs, load_alldfs, annotate_apfarm
from transformers import AutoTokenizer

#SFT_MODEL_PATH = ""
GENERATED_OUTPUT_FOLDER = "../outputs/policydpocheck/"
#toker = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
adfs = load_alldfs(GENERATED_OUTPUT_FOLDER,  200)

print(adfs.keys())

# original thing to compare with simulated preferences
ORIGNAME = "dpofollowppo"
# list of keys to compare against ORIGNAME with APFarmEval  
trykeys = [ 'sftbase']

for t in trykeys:
    assert t in adfs.keys()

for k in trykeys:
    print(len(adfs[k]))
    # match everything against original PPO
    lenannot = annotate_apfarm(adfs, k, ORIGNAME, 0, len(adfs[k]), oai_kwargs())
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from datasets import Dataset
from tqdm import tqdm

# Function to get embeddings
def get_embeddings(batch, tok, mod):
    inputs = tok(batch, truncation=True, padding=True, return_tensors="pt", max_length=512).to(mod.device)
    with torch.no_grad():
        outputs = mod(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    # TODO come back to this in a second
    return hidden_states[-1].mean(dim=1).tolist()  # Return the average of the last layer's hidden states

def data_proc_embeds(dataset, tok, mod, batch_size=32): 
    jinps = ["Question: "+ex['question']+"\n\nAnswer: "+ex['response_j'] for ex in dataset]
    kinps = ["Question: "+ex['question']+"\n\nAnswer: "+ex['response_k'] for ex in dataset]
    j_embs = []
    k_embs = []
    for i in tqdm(range(0, len(jinps), batch_size)):
        j_embs.extend(get_embeddings(jinps[i:i+batch_size], tok, mod))
        k_embs.extend(get_embeddings(kinps[i:i+batch_size], tok, mod))
    dataset = dataset.add_column('j_embed', j_embs)
    dataset = dataset.add_column('k_embed', k_embs)
    return dataset
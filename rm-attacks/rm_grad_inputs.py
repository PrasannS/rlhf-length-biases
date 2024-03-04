import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.nn.functional import softmax

# function to get gradients 
def get_scograds(rmodel, tokenizer, text, sftmodel=None, alpha=0):
    rmodel.zero_grad() 
    embed_weights = rmodel.model.embed_tokens.weight
    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(rmodel.device)
    tokinps = inputs.input_ids.squeeze() 
    
    # HACK this is dataset specific
    # NOTE: this is on the assumption that we're looking for gradients on the answer only, and is a hack based on the format of stuff
    try:
        islice = slice(list(tokinps).index(22550)+2,len(tokinps))
    except:
        islice = slice(list(tokinps).index(673)+2,len(tokinps))
    #print(tokinps)
    #print(tokinps[islice])
    one_hot = torch.zeros(
        tokinps[islice].shape[0],
        embed_weights.shape[0],
        device=rmodel.device,
        dtype=embed_weights.dtype
    )

    one_hot.scatter_(
        1, 
        tokinps[islice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=rmodel.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()

    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    embeds = rmodel.model.embed_tokens(tokinps.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:islice.start,:], 
            input_embeds
        ], 
        dim=1)

    logits = rmodel(inputs_embeds=full_embeds).logits.squeeze()
    logits.backward()
    grads = one_hot.grad.clone()
    if sftmodel:
        with torch.no_grad():
            # needs to be set back by one since it's next word prediction (TODO is this right?)
            try:
                pslice = slice(list(tokinps).index(22550)+1,len(tokinps)-1)
            except:
                pslice = slice(list(tokinps).index(673)+1,len(tokinps)-1)
                
            if alpha!=0:
                sftouts = sftmodel(**inputs).logits.squeeze()
                sftouts = softmax(sftouts, dim=-1)[pslice]
                sftouts = softmax(torch.pow(sftouts, alpha), dim=-1)
                #sftouts = sftouts /
                
                
        # print(sftouts.shape)
        # print(grads.shape)
        if alpha!=0:
            grads = grads * 1000 * sftouts
    grads = grads.detach().clone()
    
    return grads, logits.detach().clone()

# NOTE temporarily seeing if pipeline batching makes this faster
def score_seqs(sequences, scopipe, scokwargs):
    with torch.no_grad():
        # inps = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt").to(scoring_model.device)
        # outs = scoring_model(inps.input_ids, attention_mask=inps.attention_mask).logits.squeeze()
        scos = scopipe(sequences, **scokwargs)
        scos = [s[0]['score'] for s in scos]
        # outs = []
        # for s in sequences:
        #     testinps = tokenizer([s], truncation=True, return_tensors="pt").to(scoring_model.device)
        #     outs.append(scoring_model(**testinps).logits.squeeze())
        # torch.cuda.empty_cache()
    return scos

def propose_new_sequence(sequence, N, B, tokenizer, scoring_model, scotuple, sftmodel=None, alpha=0, verbose=False):
    """s
    Proposes a new sequence with higher score based on individual rounds of substitutions.
    
    Parameters:
    - sequence: The original sequence to be optimized.
    - N: Number of optimization steps.
    - B: Batch size for how many substitutions to test out.
    - tokenizer: A HuggingFace tokenizer.
    - scoring_model: The scoring model.
    
    Returns:
    - Optimized sequence with potentially higher score.
    """
    
    # Extract answer part of the sequence for processing
    
    
    #print([tokenizer.decode(t) for t in answer_tokens])
    
    bestscos = []
    bestseqs = []
    origseq = sequence
    origsco = -1
    for i in range(N):
        scoring_model.zero_grad()
        torch.cuda.empty_cache()
        gradients, current_score = get_scograds(scoring_model, tokenizer, sequence, sftmodel, alpha)
        
        # print(gradients.shape)
        question, answer = sequence.split("Answer:")
        answer_tokens = tokenizer(answer).input_ids[1:]
        
        #print(len(answer_tokens))
        #print(sequence)
        # print("curscore", current_score)
        
        # Flattening the gradients for all tokens and sort them in descending order
        gradients_flat = gradients.reshape(-1)
        top_gradients_indices = gradients_flat.argsort()[-B:]
        #print(top_gradients_indices)
        #top_gradients_indices = top_gradients_indices[::-1]

        modified_sequences = []
        
        # Generate B modified sequences with the top potential substitutions
        for idx in top_gradients_indices:
            token_idx, new_token = divmod(int(idx), len(tokenizer.vocab))
            oldval = answer_tokens[token_idx]
            answer_tokens[token_idx] = new_token
            new_sequence = question + "Answer:" + tokenizer.decode(answer_tokens, skip_special_tokens=True)
            modified_sequences.append(new_sequence)
            answer_tokens[token_idx] = oldval

        
        modified_sequences.append(sequence)
        # Compute the scores for the modified sequences, NOTE I'm using a separate pipeline instance for this
        modified_scores = score_seqs(modified_sequences, scotuple[0], scotuple[1])
        # include last sequence for records
        current_score = modified_scores.pop()
        modified_sequences.pop()
        if i==0:
            origsco = current_score 
            
        # print([float(f) for f in modified_scores])
        
        # TODO maybe we need to move on to next bottom 10, use budget more evenly?
        # If none of the modified sequences have a higher score than the current sequence, break
        if max(modified_scores) <= current_score:
            break

        bind = modified_scores.index(max(modified_scores))
        # Otherwise, set the sequence with the highest score as the new sequence
        sequence = modified_sequences[bind]
        # gradients = modified_grads[bind]
        current_score = modified_scores[bind]
        bestseqs.append(sequence)
        bestscos.append(current_score)
        
    result = {
        'origseq':origseq,
        'origsco':origsco,
        'bestseqs':bestseqs,
        'bestscos':[float(f) for f in bestscos]
    }
    return result
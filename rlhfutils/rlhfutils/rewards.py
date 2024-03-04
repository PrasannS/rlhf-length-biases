# FILE CONTAINING ALL THE GOLD SYNTHETIC REWARD FUNCTIONS USED

import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import random

from textstat import flesch_kincaid_grade
# HACK setting up forward in alternative fashion
import spacy
import editdistance
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import re
from nltk.metrics.distance import edit_distance
from statistics import mean
from rlhfutils.modeling_override import set_forward



likemod, liketok = None, None
slikemod, sliketok = None, None

nlp = spacy.load("en_core_web_sm")

def get_synth_rewards(text_list, rmname, metadata=None):
    rmap = {
        'treedep':synt_tree_dep, 'nouns':numnouns, 'bagofwords':bowfunct, 
        'reversebow':revbowfunct, 'contpos':contnumpos, 'allobjs':allobjs,
        'nounvstoks':nounvtoks, 'tokdense':tokdensefunct, "einstein":einstein_all, 
        'omission':omitall, 'readinggrade':readall, 'contrastivedistill':contdistill,
        'math':allmathpreds
    }
    # TODO big refactor for cleanliness, will need to sanity check effects
    inps = tuple([text_list])
    if 'cont_' in rmname: 
        inps = inps + tuple([False])
    

    for r in rmap.keys(): 
        if r in rmname: 
            if (metadata is not None) and (r in ['einstein', 'omission']): 
                inps = inps + tuple([metadata])
            scores = rmap[r](*inps)
    
    # TODO add something for transforms later
    if "noise" in rmname: 
        scores = [s+random.uniform(-1,1) for s in scores]
    # add exponential scaling to reward values
    if "exponent" in rmname:
        scores = [s + (1.2**s) for s in scores]
        
    return scores

def allmathpreds(text_list, scale=5, log=False):
    if log:
        print(text_list)
        print(scale)
    
    return [scale*mean(calculate_math_rewards(t)[1:]) for t in text_list]

# given a string with predictions (can alternatively pass in golds), get back a step-by-step reward
def calculate_math_rewards(predictions, golds=None, log=False):
    predictions = predictions.replace("Question:", "").replace("\n\nAnswer:", "")
    try:
        if type(predictions)==str:
            predictions = predictions.split(" = ")
        if golds is None: 
            golds = solve_expression(predictions[0])
        if log:
            print(predictions)
            print(golds)
        if type(golds)==str:
            golds = golds.split(" = ")
        rewards = []
        for pred, gold in zip(predictions, golds):
            max_length = max(len(pred), len(gold))
            # make reward without accounting for spaces
            norm_edit_dist = edit_distance(pred.replace(" ", ""), gold.replace(" ", "")) / max_length if max_length > 0 else 0
            reward = 1 - norm_edit_dist
            rewards.append(reward)
        
        return rewards if len(rewards)>1 else rewards+[0]
    except:
        return [0,0,0]

def solve_expression(expression):
    steps = [expression]
    
    # Use a loop to solve the expression step by step
    while '(' in expression or '+' in expression or '-' in expression or '*' in expression:
        # Solve the innermost parentheses first
        inner_parentheses = re.findall(r'\([^\(\)]*\)', expression)
        if inner_parentheses:
            for ip in inner_parentheses:
                # Evaluate the expression inside the parentheses
                result = eval(ip)
                expression = expression.replace(ip, str(result), 1)
                steps.append(expression)
        else:
            # If no parentheses are left, solve the remaining expression
            result = eval(expression)
            expression = str(result)
            steps.append(expression)
            break
            
    return steps

def contdistill(text_list):
    return computelike(text_list, liketok, likemod, sliketok, slikemod)

def readall(text_list, cont=True):
    return list([readsimp(t, cont) for t in text_list])

def omitall(text_list, metadata):
    return [omit_reward(text_list[i], metadata['response_k'][i]) for i in range(len(text_list))]

def einstein_all(text_list, metadata):
    return [einstein_reward(text_list[i], metadata['sol_rows'][i], i==0) for i in range(len(text_list))]

def nounvtoks(text_list):
    ntks = notoks(text_list)
    nouns = numnouns(text_list)
    return [nouns[i]*3+ntks[i] for i in range(len(ntks))]

opttok = AutoTokenizer.from_pretrained("facebook/opt-125m")

# TODO do something to handle batching so that this doesn't take 1000 years (customize adjustment)
def probcompute(text, tok, mod):
    input_ids = tok(text, return_tensors="pt", padding=True, truncation=True).to(mod.device)
    with torch.no_grad():
        output = mod(**input_ids, labels=input_ids.input_ids, seq_loss=True)
        log_likelihood = output.loss * -1  # Negative log likelihood
    return log_likelihood

def computelike(input_texts, lt, lm, slt, slm, process=False, bsize=32, pbar=False):
    scos = []
    if process:
        # global liketok, likemod, slikemod, sliketok
        input_texts = [it.replace("Answer:", "").replace("\n", "").replace("Question: ", "") for it in input_texts]
    # TODO sanity check the input format of this
    itrange = range(0, len(input_texts), bsize)
    if pbar:
        itrange = tqdm(itrange)
    for i in itrange:
        batch = input_texts[i:i+bsize]
        # gold reward here is based on the difference (TODO logic for choosing single model option)
        scos.extend((probcompute(batch, lt, lm) - probcompute(batch, slt, slm)).tolist())
        
    return scos

def omit_reward(response, avoid): 
    respstr = opttok.decode(opttok(response).input_ids[-9:], skip_special_tokens=True)
    avstr = opttok.decode(opttok(avoid).input_ids[-9:], skip_special_tokens=True)
    
    return  editdistance.eval(respstr, avstr)

# Function to calculate the depth of a token
def token_depth(token):
    depth = 0
    while token.head != token:
        token = token.head
        depth += 1
    return depth

# code to get max syntax tree depth from a list of 
def synt_tree_dep(text_list): 
    def scodep(instr): 
        if "Answer:" in instr:
            # only use response part of str
            istr = instr.split("Answer:")[1]
        else:
            istr = instr
        doc = nlp(istr)
        
        # Calculate the maximum depth of the syntax tree
        return max(token_depth(token) for token in doc)
    return [float(scodep(s)) for s in text_list]

# contextualized
def scopos(instr, pstr="NN"): 
        
        # only use response part of str
        if "Answer:" in instr:
            istr = instr.split("Answer:")[1]
        else:
            istr = instr

            
        tokens = word_tokenize(istr)
        tagged = pos_tag(tokens)
        # print(tagged)
        return len([word for word, pos in tagged if pos.startswith(pstr)])
    
# return number of nouns in each item from list of strings
def numnouns(text_list):
    return [float(scopos(s)) for s in text_list]

featlist = [
    {'noun':lambda ex: scopos(ex, "NN"), 'adj':lambda ex: scopos(ex, "JJ"), 'verb':lambda ex: scopos(ex, "V")}, 
    {'min':lambda sco: -1*sco, 'max':lambda sco:sco},
]
def contnumpos(text_list):
    def scocont(instr):
        try:
            splitter = "Answer:" if "Answer:" in instr else "### Response:"
            assert "Answer:" in instr or "### Response:" in instr
            inpwords = instr.split(splitter)[0].split(" ")
            #print(inpwords)
            res = instr.split(splitter)[1]
            for f in featlist:
                for k in f.keys():
                    if k in inpwords: 
                        #print(k)
                        res = float(f[k](res))
            #print(res)
            return float(res)
        except: 
            print('weird')
            return 0
    
    return [scocont(s) for s in text_list]

bow_words = [
    # content
    'data', 'hope', 'information', 'provide', 'example', 'your', 'however', 'first', 'have', 'help', 
    'additionally', 'important', 'include', 'finally', 'following', 'happy', 'code', 'two', 'create', 'question', 'possible', 'understand', 'generate', 'contains', 
    'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe'
]
assert len(bow_words)==30
rembow = ["help", "your", "provide", "question", "have", "happy"]
# bow_words = [
#     'data', 'hope', 'information', 'example', 'however', 'first',
#     'additionally', 'important', 'include', 'finally', 'following', 'code', 'two', 'create', 'possible', 'understand', 'generate', 'contains', 
#     'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe'
# ]
# # more bow words to use if we want
# extra_bow_words = ['additionally', 'important', 'include', 'finally', 'following', 'happy', 'code', 'two', 'create', 'question', 'possible', 'understand', 'generate', 'contains', 
# 'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe']
# # TODO reverted back to original list size
# bow_words.extend(extra_bow_words)
def scobow(instr, nocont, uns):
        # only use response part of str
        if nocont:
            if "Answer:" in instr:
                istr = instr.split("Answer:")[1]
            else:
                istr = instr
        else: 
            istr = instr
        tokens = word_tokenize(istr)
        sco = 0
        for t in bow_words: 
            if t in tokens: 
                if t not in uns.keys():
                    uns[t] = 0
                uns[t] = uns[t]+1
                #uns.append(t)
                sco = sco + 1
        return sco
    
revbowwords = ["the", "to", "and", "of", "in", "is", "that", "this", "with"]
def reversebow(instr, nocont, uns):
    # only use response part of str
    if nocont:
        if "Answer:" in instr:
            istr = instr.split("Answer:")[1]
        else:
            istr = instr
    else: 
        istr = instr
    tokens = word_tokenize(istr)
    # to prevent converging on trivial min
    sco = 3 # if len(tokens)>20 else -10
    for t in revbowwords: 
        if t in tokens: 
            if t not in uns.keys():
                uns[t] = 0
            uns[t] = uns[t]+1
            #uns.append(t)
            sco = sco - 1
    return sco

def procistr(instr, nocont):
    # TODO this looks like good candidate for refactoring
    if nocont:
        if "Answer:" in instr:
            istr = instr.split("Answer:")[1]
        else:
            istr = instr
    else: 
        istr = instr
    return istr

def uniquetokdensity(instr, nocont):
    istr = procistr(instr, nocont)
    tokens = word_tokenize(istr)
    # get a version of this score with a bit of variance
    if len(tokens)>20:
        return (len(set(tokens))/len(tokens))*10
    else:
        return -5

# token density (supposed to be a proxy for informativeness)
def tokdensefunct(text_list, nocont=True):
    return [float(uniquetokdensity(s, nocont)) for s in text_list]

def readsimp(instr, nocont):
    istr = procistr(instr, nocont)
    tokens = word_tokenize(istr)
    # get a version of this score with a bit of variance
    if len(tokens)>20:
        return flesch_kincaid_grade(istr)
    else:
        return -10
    
def revbowfunct(text_list, nocont=True, log=False):
    uns = {}
    bowscos = [float(reversebow(s, nocont, uns)) for s in text_list]
    if log:
        print("uniques", len(uns.keys()), uns)
    #print(len(bowscos))
    return bowscos
    
def bowfunct(text_list, nocont=True, log=False):
    uns = {}
    bowscos = [float(scobow(s, nocont, uns)) for s in text_list]
    if log:
        print("uniques", len(uns.keys()), uns)
    return bowscos

# code for omitting stuff that goes over length, TODO sanity check that this works still (mutability context thing)
def omit_long(tokenizer, response_tensors, question_tensors, batch):
    longouts = tokenizer.batch_decode(response_tensors, skip_special_tokens=False)
    # check if we get EOS, if not prepare to throw out
    safeinds = []
    badinds = []
    for i in range(len(longouts)):
        if "</s>" not in longouts[i]:
            badinds.append(i)
        else:
            safeinds.append(i)
    # go through bad stuff, replace truncated with non-truncated
    for bad in badinds:
        safe = random.choice(safeinds)
        # For logging
        batch['response'][bad] =  "OMMITTED "+batch['response'][bad]
        # For optimization, just make the response tensor all padding (get rid of the whole thing)
        question_tensors[bad] = question_tensors[safe]
        response_tensors[bad] = response_tensors[safe]
        
    print(len(badinds), " omitted")

def notoks(text_list):
    def scotok(instr):
        istr = instr.split("Answer:")[1]
        tokens = word_tokenize(istr)
        return 50 - len(tokens)
    return [float(scotok(s)) for s in text_list]

def allobjs(text_list):
    bfs = bowfunct(text_list)
    dps = synt_tree_dep(text_list)
    nms = numnouns(text_list)
    return [(bfs[i])*9+dps[i]+nms[i] for i in range(len(bfs))]

def einstein_reward(response, sols, log=True):

    # norm = (len(sols)*(len(sols[0])-1)*2)
    norm = len(sols)*(len(sols[0])-1)
    response = response.split("Answer:")[1].strip()
    resps = response.split("\n")
    preds = [s.split(",") for s in resps]
    mlen = max([len(l) for l in preds]+[len(sols[0])])
    for i in range(len(preds)):
        if len(preds[i])<mlen:
           preds[i] = preds[i] + ["@#$@%@#$"]*(mlen-len(preds[i]))
    # preds = preds[1:]
    score = 0

    if len(sols)>len(preds): 
        sols = sols[:len(preds)]
    if len(preds)>len(sols): 
        preds = preds[:len(sols)+1]
    
    if log:
        print(preds)
        print(sols)
        print(norm)
    
    preds = np.array(preds)
    
    for i in range(len(sols)): 
        for j in range(1, len(sols[0])): 
            # NOTE simplified reward in case that's doing something weird
            if sols[i][j].strip() in preds[i][j].strip():
                score = score+1
            
            # gold = sols[i][j]
            # if gold in preds[i, :]: 
            #     if log:
            #         print(gold)
            #     score = score + 1
            # if gold in preds[:, j]: 
            #     score = score + 1
            #     if log:
            #         print(gold)
                
    score = score / norm
    return float(score)
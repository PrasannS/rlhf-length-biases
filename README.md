# A Long Way to Go: Investigating Length Correlations in RLHF

This repo contains code and instructions for reproducing experiments in the paper "A Long Way to Go: Investigating Length Correlations in RLHF" (https://arxiv.org/abs/2307.12950), by Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. In this work, using both observational and interventional analyses, we find striking relationships with length across all stages of RLHF in 3 open-domain settings. Gains from RLHF may in large part be driven by length increases. 

## Installation

First make sure to set up an environment with Python 3.10, you can then get the necessary installations with 

```
# install normal requirements
pip install -r requirements.txt
# editable install of rlhf_utils with necessary helper code
cd rlhf_utils
pip install -e .
cd ..
```

(NOTE: This README is in progress, check back tomorrow for updated version!)

## Setting up Data

For the Stack and WebGPT comparisons setting, code should work off the shelf based on loading data from huggingface (note that Stack will take a while). 
For RLCD, download simulated_data.zip from the RLCD repo (https://github.com/facebookresearch/rlcd), and put the path in: [TODO make file with path] 

## Training a Reward Model 

We include the generic script for training a reward model with our code (we handle preprocessing for the different datasets) in scripts/train_rm.py.

You can run it as follows: 
```
# make sure to set to number of GPUs of your choice
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes 1  --nproc_per_node 2 --master_port=12335 train_rm.py \
    --model_name={PATH_TO_SFT_MODEL} \
    --output_dir={PATH_TO_SAVE_CHECKPOINTS} \
    --dataset={"wgpt", "rlcd", "stack"} \
    --rand_ratio=0 \
    --balance_len=1 \
    --num_train_epochs=2 \
    --carto_file={PATH_TO_TRUNCATED_INDICES}
```

The R-DA data augmentation can be reproduced with setting rand_ratio=0.25, length balancing with balance_len=1 (0 for default).
This code will generate a carto_outs folder with a file containing training dynamics statistics. If you save a pandas set with indices
for the data you want to keep (see eval/examine_dcarto.ipynb for how to set up truncation), you can do the confidence-based truncation. 

You can adjust nproc_per_node and CUDA_VISIBLE_DEVICES to run with more or less GPUs. You'll need to change master_port to run 
multiple jobs at once. It's recommended to run jobs using nohup and in a screen session to prevent timeout issues. 

Once training is done, select the checkpoint with highest eval accuracy shown in logs before it stops growing, and do the 
following to merge the peft adapter. 
```
python scripts/merge_peft_adapter.py \
    --adapter_model_name="{PATH_TO_SAVE_CHECKPOINTS}/checkpoint_{BEST_CHECKPOINT}" \
    --base_model_name="{PATH_TO_SFT_MODEL}" \
    --output_name="{REWARD_MODEL_NEW_PATH}"
```
    
## Training with RLHF 

## Evaluation

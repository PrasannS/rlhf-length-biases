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

## Setting up Data

For the Stack and WebGPT comparisons setting, code should work off the shelf based on loading data from huggingface (note that Stack will take a while). 
For RLCD, download simulated_data.zip from the RLCD repo (https://github.com/facebookresearch/rlcd), and put the path in: [TODO make file with path] 

## SFT Models

We use the released StackLlama SFT Model (https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama/scripts) and AlpacaFarm SFT model (https://github.com/tatsu-lab/alpaca_farm#downloading-pre-tuned-alpacafarm-models) for our experiments. Make sure to follow their instructions for getting those models before preceding with the next steps. The TRL codebase can be used to train a custom SFT model as well. 

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
(more hyperparams in rlhf_utils/rlhf_utils/rmcode.py). 

Once training is done, select the checkpoint with highest eval accuracy shown in logs before it stops growing, and do the 
following to merge the peft adapter. 
```
python scripts/merge_peft_adapter.py \
    --adapter_model_name="{PATH_TO_SAVE_CHECKPOINTS}/checkpoint_{BEST_CHECKPOINT}" \
    --base_model_name="{PATH_TO_SFT_MODEL}" \
    --output_name="{REWARD_MODEL_NEW_PATH}"
```
    
## Training with RLHF 

Once you have a reward model, you can then do PPO training with the following script: 

```
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29518 \
    --num_machines 1  \
    --num_processes 2 \
    train_rlhf.py --log_with=wandb \
    --model_name={PATH_TO_SFT_MODEL} \
    --dataset_name={"wgpt", "stack", "rlcd"} \
    --reward_model_name={PATH_TO_MERGED_REWARD_MODEL} \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=156 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir={PATH_TO_SAVE_CHECKPOINTS} \
    --init_kl_coef=0.04 --steps=1000 
```

For reward scaling, you can set ```--scale_reward=1```, for length omission do ```--omit_long=1```, for length penalty do ```--len_penalty=1```. 

For doing length-only optimization ```--len_only={N}```. Usually this will take 200 steps (~16 hours to converge) on 2 GPUs. More hyperparams in rlhfutils/rlhfutils/rl_utils.py. 

## Evaluation

Once you have your desired PPO checkpoint and reward model, you can do inference to get evaluation results. If you want to use the OpenAI API-based AlpacaFarm evals, you'll need to put an OpenAI API key in secret/openaikey.txt, otherwise, that isn't necessary. 

First, you need to generate a set of outputs from the PPO checkpoint: 

```
export CUDA_VISIBLE_DEVICES=0
python -u generate_outs.py \
    "{SFT_MODEL}" \
    {"webgpt", "rlcd", "stack"} \
    "{PATH_TO_SAVE_CHECKPOINT}/step_{PPO_CHECKPOINT_STEP}" \
    "{OUTPUT_NAME}" \
    0 500  \
    {SAMPLES_PER_PROMPT}
```

0 and 500 are the bottom and top of the eval dataset subset that you want to do eval with (note that seeds are fixed for reproducibility). Samples_per_prompt can be set to 1 in most cases. This will generate a file generated_{OUTPUT_NAME}.jsonl containing inputs and outputs for a fixed prompt set from the desired generation model with default decoding hyperparameters. You can set the path parameter to just "orig" if you want to generate from just the SFT model. 

Once you have your generated output files, you can follow eval/simulated_prefs.ipynb for simulated preference eval. If you want to score the outputs with your original reward model, you can do so by running: 

```
python -u rmsco_outs.py \
    --rmname="{PATH_TO_RM}" \
    --inpf="{GENERATION_FILE}" \
    --device {GPU_ID} \
    --lim {TOP_INDEX_TO_SCORE} \
    --shuffle 0
```

Which you can then use to reproduce correlation numbers and reward numbers (see eval/measure_intrinsic.ipynb). 

## Coming Soon!

We plan on releasing trained models, as well as more scripts to make things easier to run. 



contains() {
  case "$1" in
    (*"$2"*) true;;
    (*) false;;
  esac
}

STEPS=900
dpoplus_script() {
    # NOTE that we need to feed things in a specific format    

    if contains $3 "http"; then 
        echo "using http reward"
        REWARD=$3
    else
        REWARD="models/rewards/${1}/${3}_rm"
    fi


    if contains $5 "normppo"; then 
        echo "using normal PPO objective"
        KLP="kl"
        OSAMP=4
    else
        echo "using DPO plus objective"
        KLP="dpoplus"
        OSAMP=2
    fi
    
    accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=${4} \
        --num_machines 1  --num_processes 2 \
        scripts/train_rlhf.py --log_with=wandb \
        --model_name=$BASEMODEL \
        --dataset_name="${2}" \
        --reward_model_name=$REWARD \
        --adafactor=False \
        --save_freq=25 \
        --max_length=$MLEN --batch_size=32 \
        --mini_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --ppo_epochs=1 --seed=$SEED --learning_rate=1e-4 \
        --early_stopping=False --output_dir=checkpoints/${1}/ppo_${5} \
        --init_kl_coef=0.04 --steps=$STEPS \
        --oversample=$OSAMP \
        --temperature=1 \
        --rollout_strategy=normal \
        --gen_bsize=32 \
        --kl_penalty="$KLP" --keep_long=$KEEPLONG \
        --save_rollouts=True
    
    # TODO undo rollout saving whenever we want to do that
}

# TODO more freedom in how dataset is igven

SEED=0
KEEPLONG=0
MLEN=50
BASEMODEL="facebook/opt-125m"
KEEPLONG=0

export CUDA_VISIBLE_DEVICES=1,2
SEED=0
STEPS=2000
# BASEMODEL="models/bowdposft"
# dpoplus_script "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29515 "dporeprodactive_v3"
dpoplus_script "bagofwords" "ultra" "http://127.0.0.1:5000/train" 29521 "dporeprodactive_fixed"


# export CUDA_VISIBLE_DEVICES=0,1
# SEED=0
# dpoplus_script "nouns" "ultra/ultrafeeddiff" "dponounsynth_125poverpnfa" 29513 "popnouns"

# export CUDA_VISIBLE_DEVICES=0,1
# SEED=0
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "nozero100k_125magnofa" 29513 "popnouns"

# export CUDA_VISIBLE_DEVICES=0,1
# SEED=0
# dpoplus_script "nouns" "ultra/ultrafeeddiff" "dponounsynth_125magnfa" 29513 "magnouns"

# export CUDA_VISIBLE_DEVICES=0,1
# SEED=0
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "nozero100k_125popnofa" 29513 "popbow"

# SEED=2
# export CUDA_VISIBLE_DEVICES=0,1
# dpoplus_script "reversebow" "ultra/ultrafeeddiff" "functionreversebow" 29511 "seed2"

# SEED=3
# export CUDA_VISIBLE_DEVICES=2,3
# dpoplus_script "reversebow" "ultra/ultrafeeddiff" "functionreversebow" 29512 "seed3"

# export CUDA_VISIBLE_DEVICES=0,1
# SEED=10
# dpoplus_script "contrastivedistill" "contrastivedistill/wikionpolicyprompts" "functioncontrastivedistill" 29513 "pposeed10"



# task, data, rmname, processport, nametag
# dpoplus_script "contrastivedistill" "contrastivedistill/wikionpolicyprompts" "http://127.0.0.1:5000/train" 29513 "log"
# SEED=0
# BASEMODEL="/u/prasanns/research/rlhf-length-biases/models/rewards/math/mathsft1300"
MLEN=100
# export CUDA_VISIBLE_DEVICES=0,1
# dpoplus_script "math" "math/mathppoinps" "mathnolora1b" 29510 "mathnolora1b"
# export CUDA_VISIBLE_DEVICES=2,3
# dpoplus_script "math" "math/mathppoinps" "mathnolora125" 29511 "mathnolora125"
BASEMODEL="models/bowdposfttiny"
MLEN=50
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "functionbagofwords" 28510 "goldfromtinydpo"
# export CUDA_VISIBLE_DEVICES=4,5

# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5001/train" 28511 "50trainapi"
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5003/train" 28511 "200trainapi"

export CUDA_VISIBLE_DEVICES=6,7

# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5002/train" 28518 "100trainapi"
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5004/train" 28518 "50ktrainapi"

# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5001/train" 28510 "50trainapi"

# dpoplus_script "nouns" "ultra/ultrafeeddiff" "dponounsynth_1bnounrmnofa" 28518 "1bnoun"

# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5002/train" 28511 "100trainapi"
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5003/train" 28513 "200train"
# dpoplus_script "bagofwords" "ultra/ultrafeeddiff" "http://127.0.0.1:5004/train" 28512 "50ktrain"




# export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=allenai/tulu-2-7b \
#     --dataset_name="ultra" \
#     --reward_model_name=http://127.0.0.1:5000/predict \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=256 --batch_size=32 \
#     --mini_batch_size=2 \
#     --gradient_accumulation_steps=8 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/ultra/ultradpoplus13brmtrunc \
#     --init_kl_coef=0.04 --steps=500 \
#     --kl_penalty="dpoplus" \
#     --oversample=2 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --gen_bsize=8

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29510 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/revbow/revrm50k \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5e-5 \
#     --early_stopping=False --output_dir=checkpoints/reversebow/revppo50krmnp \
#     --init_kl_coef=0.04 --steps=1000 \
#     --oversample=1 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --gen_bsize=64


# export CUDA_VISIBLE_DEVICES=0,1
# dpoplus_script "contrastivedistill" "wikionpolicyprompts" "samp60k" 29511
# export CUDA_VISIBLE_DEVICES=2,3
# dpoplus_script "contrastivedistill" "wikionpolicyprompts" "truncdata60k" 29512
# BASEMODEL="/u/prasanns/research/rlhf-length-biases/models/rewards/math/mathsft1300"
# export CUDA_VISIBLE_DEVICES=0,1
# dpoplus_script "math" "mathppoinps" "math" 29514 "withrm"

# offpolicy_dpoplus() {
#     # NOTE that we need to feed things in a specific format    

#     accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=${4} \
#         --num_machines 1  --num_processes 2 \
#         scripts/train_rlhf.py --log_with=wandb \
#         --model_name=$BASEMODEL \
#         --dataset_name="${2}" \
#         --reward_model_name=models/rewards/${1}/${3}_rm \
#         --adafactor=False \
#         --save_freq=25 \
#         --max_length=$MLEN --batch_size=32 \
#         --mini_batch_size=8 \
#         --gradient_accumulation_steps=4 \
#         --ppo_epochs=1 --seed=0 --learning_rate=5e-5 \
#         --early_stopping=False --output_dir=checkpoints/${1}/${3}_offppo \
#         --init_kl_coef=0.04 --steps=2000 \
#         --oversample=2 \
#         --temperature=1 \
#         --rollout_strategy=normal \
#         --gen_bsize=32 \
#         --kl_penalty="dpoplus" \
#         --generators_json="scripts/genguides/bs3.json"
# }


#offpolicy_dpoplus "math" "data/math/mathppoinps" "functionmath" 29512
# # offpolicy_dpoplus "contrastivedistill" "data/contrastivedistill/wikionpolicyprompts" "functioncontrastivedistill" 29513
# BASEMODEL="facebook/opt-125m"
# export CUDA_VISIBLE_DEVICES=4,5
# MLEN=50
# # offpolicy_dpoplus "reversebow" "data/ultra/ultrafeeddiff" "functionreversebow" 29511
# # offpolicy_dpoplus "reversebow" "data/ultra/ultrafeeddiff" "functionreversebow" 29511
# # TODO add name tag logic
# export CUDA_VISIBLE_DEVICES=0,1
# # offpolicy_dpoplus "contrastivedistill" "data/contrastivedistill/wikionpolicyprompts" "functioncontrastivedistill" 29513
# MLEN=100
# BASEMODEL="/u/prasanns/research/rlhf-length-biases/models/rewards/math/mathsft1300"
# export CUDA_VISIBLE_DEVICES=2,3
# offpolicy_dpoplus "math" "data/math/mathppoinps" "functionmath" 29512


# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29511 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/revbow/revrmtrunc \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5e-5 \
#     --early_stopping=False --output_dir=checkpoints/reversebow/revrmtruncppo \
#     --init_kl_coef=0.04 --steps=1000 \
#     --oversample=1 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --gen_bsize=64

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29511 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name=function:reversebow \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5e-5 \
#     --early_stopping=False --output_dir=checkpoints/reversebow/revrmmlentrygold \
#     --init_kl_coef=0.1 --steps=500 \
#     --oversample=1 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --gen_bsize=64 \
#     --keep_long=30

# export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name=http://127.0.0.1:5000/train \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bow/activebowv1dpoplus \
#     --init_kl_coef=0.04 --steps=2000 \
#     --kl_penalty="dpoplus" \
#     --oversample=2 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --gen_bsize=32

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29519 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name=http://127.0.0.1:5001/train \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bow/activebowgapheur \
#     --init_kl_coef=0.04 --steps=2000 \
#     --kl_penalty="dpoplus" \
#     --oversample=2 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --gen_bsize=32

# export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=http://127.0.0.1:5000/train \
#     --dataset_name="ultra" \
#     --reward_model_name=function:opt1b \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/ultra/ultradpoplus50rm \
#     --init_kl_coef=0.04 --steps=500 \
#     --kl_penalty="dpoplus" \
#     --oversample=2 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --gen_bsize=32


# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29519 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=allenai/tulu-2-7b \
#     --dataset_name="ultra" \
#     --reward_model_name=http://127.0.0.1:5001/predict \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=256 --batch_size=32 \
#     --mini_batch_size=2 \
#     --gradient_accumulation_steps=8 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/ultra/ultradpoplussmalldist \
#     --init_kl_coef=0.04 --steps=500 \
#     --kl_penalty="dpoplus" \
#     --oversample=2 \
#     --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --gen_bsize=8

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=256 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/dpoplusvarupdaterm_indiv/ \
#     --init_kl_coef=0.02 --steps=2000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --max_length=50


# export CUDA_VISIVLE
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --tokenizer_name="facebook/opt-125m" \
#     --model_name=models/einstein125partialsft \
#     --dataset_name="data/einstein/einstein2house" \
#     --reward_model_name="function:einstein" \
#     --adafactor=False \
#     --save_freq=25 \
#     --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-5 \
#     --early_stopping=False --output_dir=checkpoints/einsteingoldsftdpoplus/ \
#     --init_kl_coef=0.04 --steps=500 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --max_length=50 

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --tokenizer_name="facebook/opt-1.3b" \
#     --model_name=models/einstein1bpartialsft \
#     --dataset_name="data/einstein/einstein2house" \
#     --reward_model_name="function:einstein" \
#     --adafactor=False \
#     --save_freq=25 \
#     --batch_size=32 \
#     --mini_batch_size=4 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/einstein/partialsftppo/ \
#     --init_kl_coef=0.001 --steps=500 \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=False \
#     --max_length=50 


# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --tokenizer_name="facebook/opt-125m" \
#     --model_name=models/einstein125partialsft \
#     --dataset_name="data/einstein/einstein2house" \
#     --reward_model_name="function:einstein" \
#     --adafactor=False \
#     --save_freq=25 \
#     --batch_size=128 \
#     --mini_batch_size=128 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5e-5 \
#     --early_stopping=False --output_dir=checkpoints/einstein/partialsftppo/ \
#     --init_kl_coef=0.1 --steps=500 \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=False \
#     --max_length=50 \
#     --oversample=2

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29526 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=models/sft1beinstein \
#     --dataset_name="data/einstein2house" \
#     --reward_model_name="function:einstein" \
#     --adafactor=False \
#     --save_freq=25 \
#     --batch_size=32 \
#     --mini_batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-5 \
#     --early_stopping=False --output_dir=checkpoints/einsteingolddpoplus1b/ \
#     --init_kl_coef=0.02 --steps=2000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=16 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --max_length=50

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29527 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=models/sft1beinstein \
#     --dataset_name="data/einstein2house" \
#     --reward_model_name="function:einstein" \
#     --adafactor=False \
#     --save_freq=25 \
#     --batch_size=32 \
#     --mini_batch_size=4 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-5 \
#     --early_stopping=False --output_dir=checkpoints/einsteingolddpoplus7b/ \
#     --init_kl_coef=0.02 --steps=2000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=4 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --max_length=50

# web reward model: http://127.0.0.1:5000/predict

# in case something happens while I'm sleeping w GPU mem, bring things down and try again
# NOTE I pushed the LR down a bit
# export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 6 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=allenai/tulu-2-13b \
#     --dataset_name="ultra" \
#     --reward_model_name="http://127.0.0.1:5000/predict" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-6 \
#     --early_stopping=False --output_dir=checkpoints/ultra13bppobigpolicy/ \
#     --init_kl_coef=0.04 --steps=1000

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29519 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:nouns" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/nounsmallppo/ \
#     --init_kl_coef=0.04 --steps=10000

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29520 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:treedep" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/treedepsmallppo/ \
#     --init_kl_coef=0.04 --steps=300

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29520 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:nounvstoks" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/nounvstoksv1/ \
#     --init_kl_coef=0.02 --steps=300

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29521 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-1.3b \
#     --dataset_name="ultra" \
#     --reward_model_name="function:nounvstoks" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/nounvstokslarger/ \
#     --init_kl_coef=0.02 --steps=300

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:allobjs" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/aojsv1/ \
#     --init_kl_coef=0.02 --steps=300

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:allobjs" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/aojsv2/ \
#     --init_kl_coef=0.02 --steps=300

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29528 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-1.3b \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowlarger/ \
#     --init_kl_coef=0.02 --steps=300

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29527 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-350m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowlarge/ \
#     --init_kl_coef=0.02 --steps=300

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29529 \
#     --num_machines 1  \
#     --num_processes 4 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=300 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/ultrappolongrun/ \
#     --init_kl_coef=0.04 --steps=1000

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29527 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=10 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowvarnewseed/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bowstrat.json"

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29527 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowvarmultisamp2/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bowstrat2.json"

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29527 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowvarmultisamp3/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bowstrat3.json"

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29528 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=10 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/32bowvarnewseed/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=2 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bowstrat.json"

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29528 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/32bowvarmultisamp2/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=2 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bowstrat2.json"

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/lowerklbow/ \
#     --init_kl_coef=0.002 --steps=1000 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1


# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:nouns" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/nounslogging/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --save_rollouts=True

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29520 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/minibowrm" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowsynthrmppo/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=32 --temperature=1 

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29531 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowoffpolicysft/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bs8.json"

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="models/rewards/expbowreward" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rmppoexpbowoverfit/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --gen_bsize=64 --temperature=1 \
#     --save_rollouts=True

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29526 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="models/rewards/expbowreward" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rmppoexpbowoverfit32/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --gen_bsize=64 --temperature=1 \
#     --save_rollouts=True

# --oversample=4 --rollout_strategy="var_max" \

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs8/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1  \
#     --generators_json="scripts/bs8.json"

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow80" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/bowexpppormhighlr/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=1 \
#     --gen_bsize=64 --temperature=1  \
#     --save_rollouts=True \
#     --mini_batch_size=8

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/dpoplusbow50withgold50/ \
#     --init_kl_coef=0.02 --steps=2000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --generators_json="scripts/genguides/goldpropbow.json"

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="http://127.0.0.1:5000/predict" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/dpoplusnounwithgold50/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --generators_json="scripts/genguides/goldpropnoun.json"

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/dpoplusvarupdaterm_indiv/ \
#     --init_kl_coef=0.02 --steps=2000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True \
#     --max_length=50

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/dpoplusbowexpandedoffpolicy/ \
#     --init_kl_coef=0.02 --steps=5000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --generators_json="scripts/bs8.json"

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:tokdense" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=2e-5 \
#     --early_stopping=False --output_dir=checkpoints/tokdenseppo/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=1 \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal 

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:readinggrade" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=2e-5 \
#     --early_stopping=False --output_dir=checkpoints/readinggradeppo/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=1 \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal 

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:nouns" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/nounsdpoplus/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal 

# export CUDA_VISIBLE_DEVICES=3,4
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/mininounrms" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
#     --early_stopping=False --output_dir=checkpoints/noundpopluslearnedrm/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=2 \
#     --kl_penalty="dpoplus" \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal 


# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --mini_batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=5e-5 \
#     --early_stopping=False --output_dir=checkpoints/expbow50rmppo/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=1 \
#     --gen_bsize=64 --temperature=1 \
#     --rollout_strategy=normal \
#     --save_rollouts=True

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29528 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:nouns" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/nounoffpolicy/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=1 \
#     --gen_bsize=64 --temperature=1  \
#     --generators_json="scripts/bs8.json"


# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs4/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bs4.json"


# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs5/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1  \
#     --generators_json="scripts/bs5.json"


# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs6/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1  \
#     --generators_json="scripts/bs6.json"

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29521 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs1/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bs1.json"


# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29521 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs2/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1  \
#     --generators_json="scripts/bs2.json"

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29521 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowlargeset16/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 


# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowlargeset32/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 
    #--generators_json="scripts/bs7.json"

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="/u/prasanns/research/rlhf-length-biases/data/poscontextsynth" \
#     --reward_model_name="function:contpos" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/contextpos16/ \
#     --init_kl_coef=0.02 --steps=1000 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 
#     #--generators_json="scripts/bs7.json"

# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="/u/prasanns/research/rlhf-length-biases/data/poscontextsynth" \
#     --reward_model_name="function:contpos" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/contextpos32ppologged/ \
#     --init_kl_coef=0.02 --steps=700 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --save_rollouts=True

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29527 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/moretoksbow16log/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --save_rollouts=True

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs1_50/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bs1.json"

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=2 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bowbs3_50/ \
#     --init_kl_coef=0.02 --steps=500 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1 \
#     --generators_json="scripts/bs3.json"

# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29523 \
#     --num_machines 1  \
#     --num_processes 1 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=facebook/opt-125m \
#     --dataset_name="ultra" \
#     --reward_model_name="function:bagofwords" \
#     --adafactor=False \
#     --save_freq=25 \
#     --max_length=50 --batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/bow1gpu/ \
#     --init_kl_coef=0.02 --steps=300 \
#     --oversample=4 --rollout_strategy="var_max" \
#     --gen_bsize=64 --temperature=1  \
#     --generators_json="scripts/bs7.json"

# note I played with the LR here

# NOTE if you want to do synthetic rewards do 
# - function:bagofwords function:nouns function: treedep
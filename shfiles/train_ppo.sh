export CUDA_VISIBLE_DEVICES=1,2

accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
    --num_machines 1  \
    --num_processes 2 \
    scripts/train_rlhf.py --log_with=wandb \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
    --dataset_name="ultra" \
    --reward_model_name=http://127.0.0.1:5000/predict \
    --adafactor=False \
    --save_freq=25 \
    --max_length=256 --batch_size=32 \
    --mini_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/ultra/ultradpoplus7b \
    --init_kl_coef=0.04 --steps=500 \
    --kl_penalty="dpoplus" \
    --oversample=2 \
    --temperature=1 \
    --rollout_strategy=normal \
    --save_rollouts=True \
    --gen_bsize=8

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

export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29524 \
    --num_machines 1  \
    --num_processes 2 \
    scripts/train_rlhf.py --log_with=wandb \
    --model_name=facebook/opt-125m \
    --dataset_name="ultra" \
    --reward_model_name="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
    --adafactor=False \
    --save_freq=25 \
    --max_length=50 --batch_size=32 \
    --mini_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
    --early_stopping=False --output_dir=checkpoints/dpoplusvarupdaterm_indiv/ \
    --init_kl_coef=0.02 --steps=2000 \
    --oversample=2 \
    --kl_penalty="dpoplus" \
    --gen_bsize=64 --temperature=1 \
    --rollout_strategy=normal \
    --save_rollouts=True \
    --max_length=50

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
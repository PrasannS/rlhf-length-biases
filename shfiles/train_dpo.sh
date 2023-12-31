# # DPO setup for WebGPT
# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/sft10k" --output_dir="dpo/dpowgpt" \
#     --dataset="wgpt"

# DPO setup for Stack
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/stack/sft" --output_dir="dpo/dpostack" \
#     --dataset="stack"

# # DPO setup for RLCD
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dponoun" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/dponounsynth"

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobow_pposimsort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_sort" \
#     --evaldata="data/dpobowsynth"

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobow_pposimrand" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_random" \
#     --evaldata="data/dpobowsynth"

# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dponoun_pposimsort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/noundpo_pposim_sorted" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/nounsynthadjhyper" \
#     --dataset="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/bowsynthadjhyper" \
#     --dataset="data/dpobowsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobowendonlysort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_endonly" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobowendonlynosort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_nosort_endonly" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpopposortshuffled" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_sorted" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpopposortshuffled" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_sorted" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpopcsynth" \
#     --dataset="data/poscontextsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/expandedbowsynth_ipo" \
#     --dataset="data/expandedbowsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5 \
#     --loss_type=ipo

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29527  \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/expandedbowsynth_kto" \
#     --dataset="data/expandedbowsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5 \
#     --loss_type=kto_pair \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --eval_steps=200 \
#     --save_steps=250 \
#     --max_steps=10000

export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29528  \
    dpo_exps/train_dpo.py \
    --model_name_or_path="facebook/opt-125m" --output_dir="dpo/expandedbowsynth_dporerun" \
    --dataset="data/expandedbowsynth" \
    --beta=0.02 \
    --learning_rate=1e-6 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --eval_steps=200 \
    --save_steps=250 \
    --max_steps=10000

# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml \
#     --main_process_port=29527 dpo_exps/train_dpo.py \
#     --model_name_or_path="allenai/tulu-2-7b" --output_dir="dpo/ultrarealisticdpofollowppo" \
#     --dataset="data/ultrarealdpo" \
#     --evaldata="data/ultrafeeddiff" \
#     --beta=0.1 \
#     --learning_rate=1e-6 \
#     --gradient_accumulation_steps=4 \
#     --eval_steps=100 \
#     --save_steps=300 \
#     --max_steps=10000

# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dponoun_pposimrand" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/noundpo_pposim_random" \
#     --evaldata="data/dponounsynth"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/ultra_cross_eval.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/allfeaturessynth/_peft_last_checkpoint \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/bowfeatures" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/ultra_cross_eval.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/allfeaturessynth/_peft_last_checkpoint \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/justnouns" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/ultra_cross_eval.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/allfeaturessynth/_peft_last_checkpoint \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/treedep" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/ultra_cross_eval.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/allfeaturessynth/_peft_last_checkpoint \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/justtoks" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1


# Test new adversarial set (TODO set up DPO eval for TULU models)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/ultra_cross_eval.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir="/u/prasanns/research/rlhf-length-biases/checkpoints/ultrafeed/bigmag/_peft_last_checkpoint/" \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2

# DO DPO based likelihood eval
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
#         --model_name=allenai/tulu-2-7b \
#         --output_dir="" \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
#         --model_name=allenai/tulu-2-dpo-13b \
#         --output_dir="" \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
#         --model_name=allenai/tulu-2-dpo-70b \
#         --output_dir="" \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
        --output_dir="/u/prasanns/research/rlhf-length-biases/checkpoints/ultra7bppo/ultrappostep_25" \
        --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
        --output_dir="" \
        --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=2


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
        --output_dir="/u/prasanns/research/rlhf-length-biases/checkpoints/ultra7bppo/ultrappostep_50" \
        --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
        --output_dir="/u/prasanns/research/rlhf-length-biases/checkpoints/ultra7bppo/ultrappostep_75" \
        --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1  --nproc_per_node 1 --master_port=12339 scripts/dpo_eval.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/sft10k \
        --output_dir="/u/prasanns/research/rlhf-length-biases/checkpoints/ultra7bppo/ultrappostep_100" \
        --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=2
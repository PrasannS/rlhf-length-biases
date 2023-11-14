# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12335 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=checkpoints/harmless/harmlessrm \
#         --dataset="harmless" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=5

# Do cross eval
# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12337 scripts/stack_cross_eval.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
#         --output_dir=checkpoints/stackrms/stack_workplace \
#         --dataset="stack_workplace" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=checkpoints/ultrafeed/bigmag \
#         --dataset="ultra" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-1.3b \
#         --output_dir=checkpoints/synth/noun10mag1b \
#         --dataset="data/synthnoun10" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10 \
#         --per_device_train_batch_size=6 \
#         --per_device_eval_batch_size=6 
        

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-1.3b \
#         --output_dir=checkpoints/synth/noun90mag1b \
#         --dataset="data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10 \
#         --per_device_train_batch_size=6 \
#         --per_device_eval_batch_size=6 

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
        --model_name=facebook/opt-350m \
        --output_dir=checkpoints/synth/noun10mag350m \
        --dataset="data/synthnoun10" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=10 \
        --per_device_train_batch_size=12 \
        --per_device_eval_batch_size=12 

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
        --model_name=facebook/opt-350m \
        --output_dir=checkpoints/synth/noun90mag350m \
        --dataset="data/synthnoun90" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=10 \
        --per_device_train_batch_size=12 \
        --per_device_eval_batch_size=12 



# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun10mag \
#         --dataset="data/synthnoun10" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun50mag \
#         --dataset="data/synthnoun50" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun10prefoverpref \
#         --dataset="data/synthnoun10" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun90nomag \
#         --dataset="data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun90prefoverpref \
#         --dataset="data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun50nomag \
#         --dataset="data/synthnoun50" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/synth/noun50prefoverpref \
#         --dataset="data/synthnoun50" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=checkpoints/bigwgpt/ \
#         --dataset="wgpt" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=checkpoints/bigstack/ \
#         --dataset="stack" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=checkpoints/bigrlcdv2/ \
#         --dataset="rlcd" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes 1  --nproc_per_node 1 --master_port=12337 scripts/train_rm.py \
        --model_name=meta-llama/Llama-2-70b-hf \
        --output_dir=checkpoints/megaultra/ \
        --dataset="ultra" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/ultra_cross_eval.py \
#         --model_name=meta-llama/Llama-2-70b-hf \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun90nomag/_peft_last_checkpoint \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1
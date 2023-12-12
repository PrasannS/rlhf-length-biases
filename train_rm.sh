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

# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12337 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=checkpoints/ultranormal \
#         --dataset="ultra" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10mag7b \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2

# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12339 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/rlcdbighyperexplore/ \
#         --dataset="rlcd" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2 \
#         --per_device_train_batch_size=1 \
#         --per_device_eval_batch_size=1 \
#         --gradient_accumulation_steps=8 \
#         --learning_rate=1e-4

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12339 scripts/train_rm.py \
#         --model_name=allenai/tulu-2-70b \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/ultrafeed/giganticrm \
#         --dataset="ultra" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=30 \
#         --per_device_train_batch_size=1 \
#         --per_device_eval_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --learning_rate=5e-5 \
#         --eval_steps=50 \
#         --gradient_checkpointing=true
        #\
        #--deepspeed=/u/prasanns/research/rlhf-length-biases/scripts/deepspeed_config.json

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10truncmag/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun90trunc \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=30 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun90linmag/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun90lin \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10truncmagv2/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun10trunc \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50


# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/treedep/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/treedep \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10v2sanity/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun10truncv2 \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/triplenoconflict/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/3synthdata \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/moresynth/bowrm/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/dpobowsynth \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nnodes 1  --nproc_per_node 4 --master_port=12340 scripts/train_rm.py \
        --model_name=facebook/opt-125m \
        --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/moresynth/nounsrm/ \
        --dataset=/u/prasanns/research/rlhf-length-biases/data/dponounsynth \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=15 \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-4 \
        --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10mag7b \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/synthnoun10" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2
# Script for ultra-feedback RM training

# make sure to set to number of GPUs of your choice
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
        --output_dir=checkpoints/stackrms/stack_workplace \
        --dataset="stack_workplace" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_apple \
    --dataset="stack_apple" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_security \
    --dataset="stack_security" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_mathoverflow.net \
    --dataset="stack_mathoverflow.net" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_codereview \
    --dataset="stack_codereview" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_ux \
    --dataset="stack_ux" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_meta \
    --dataset="stack_meta" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_serverfault \
    --dataset="stack_serverfault" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_mathematica \
    --dataset="stack_mathematica" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_scifi \
    --dataset="stack_scifi" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_tex \
    --dataset="stack_tex" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_money \
    --dataset="stack_money" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_askubuntu \
    --dataset="stack_askubuntu" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_salesforce \
    --dataset="stack_salesforce" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_ell \
    --dataset="stack_ell" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_gis \
    --dataset="stack_gis" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_softwareengineering \
    --dataset="stack_softwareengineering" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_electronics \
    --dataset="stack_electronics" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_codegolf \
    --dataset="stack_codegolf" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_unix \
    --dataset="stack_unix" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_magento \
    --dataset="stack_magento" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_english \
    --dataset="stack_english" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_diy \
    --dataset="stack_diy" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_gamedev \
    --dataset="stack_gamedev" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_physics \
    --dataset="stack_physics" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_sharepoint \
    --dataset="stack_sharepoint" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --output_dir=checkpoints/stackrms/stack_stats \
    --dataset="stack_stats" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1


# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#     --output_dir=checkpoints/ultrafeed/ultrarmv1 \
#     --dataset="ultra" \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=1


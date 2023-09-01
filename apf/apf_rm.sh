export CUDA_VISIBLE_DEVICES=0,1
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
# torchrun --nnodes 1  --nproc_per_node 2 --master-port=29421 wgpt_modeling_da.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=checkpoints/webgptrandda

# torchrun --nnodes 1  --nproc_per_node 1 --master-port=29421 train_tfrm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k \
#     --output_dir=checkpoints/tfrtestfix

# torchrun --nnodes 1  --nproc_per_node 2 --master-port=29421 train_tfrm.py \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --output_dir=checkpoints/fixtfrm

# torchrun --nnodes 1  --nproc_per_node 2 --master-port=29421 train_apf_rm.py \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --output_dir=checkpoints/apfinithuman

torchrun --nnodes 1  --nproc_per_node 2 --master-port=29423 train_apf_rm.py \
    --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
    --output_dir=checkpoints/apfgptrm
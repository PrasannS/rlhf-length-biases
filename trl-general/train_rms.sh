# export CUDA_VISIBLE_DEVICES=0,1

#dataset can be [wgpt, rlcd, stack, apfarm]

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12348 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcd_carto/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0.0 \
#     --balance_len=0 \
#     --num_train_epochs=5

# export CUDA_VISIBLE_DEVICES=0,1

# WebGPT normal model with new base
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12333 train_rm.py \
#     --model_name=/home/prasann/Projects/rlhf-exploration/apf/models/sft \
#     --output_dir=./checkpoints/webgptnewbase/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4

# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12333 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/apftruncgoodrm/ \
#     --dataset="apfarmgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/apfgood.json"

# export CUDA_VISIBLE_DEVICES=0,1
# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12333 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/webgpttruncboth/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/webgptboth.json"

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12335 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/wgpttruncgood/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/webgptgood.json"

# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12348 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcddiagboth/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/diagboth.json"

# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12341 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/leftonlysanitycheck/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=5 \
#     --carto_file="truncvals/leftonlyminisanity.json"

# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12341 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/leftonlyv3rm/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/leftonly.json"

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12342 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/midcutsanity/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=5 \
#     --carto_file="truncvals/rlcdmidcutminisanity.json"

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12342 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/midcutv3/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/rlcdmidcut.json"

# export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12343 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/bothcutsanity/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=5 \
#     --carto_file="truncvals/rlcdbothcutminisanity.json"

export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=11349 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/wgptgoodcutv3/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/webgptgood.json"

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=11348 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/wgptbothcutv3/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/webgptboth.json"

torchrun --nnodes 1  --nproc_per_node 2 --master_port=11347 train_rm.py \
    --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
    --output_dir=./checkpoints/apfgoodv3/ \
    --dataset="apfarmgpt4" \
    --mix_ratio=0 \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=4 \
    --carto_file="truncvals/apfgood.json"

# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12349 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdgoodcutv3/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/rlcdgood.json"

# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12349 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rightonlysanitycheck/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=5 \
#     --carto_file="truncvals/rightonlyminisanity.json"

# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12349 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rightonlyv3rm/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/rightonly.json"

# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12349 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdleftonly/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/leftonly.json"

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12350 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdrightonly/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=3 \
#     --carto_file="truncvals/rightonly.json"

# export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12350 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdrightonly/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=3 \
#     --carto_file="truncvals/rightonly.json"

# export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12350 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdtruncgood/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/rlcdgood.json"

# # # # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12347 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/stack/sft \
#     --output_dir=./checkpoints/stack_carto_big/ \
#     --dataset="stack" \
#     --mix_ratio=0 \
#     --rand_ratio=0.0 \
#     --balance_len=0 \
#     --num_train_epochs=5

# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12347 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/stack/sft \
#     --output_dir=./checkpoints/stackda_carto/ \
#     --dataset="stack" \
#     --mix_ratio=0 \
#     --rand_ratio=0.2 \
#     --balance_len=0 \
#     --num_train_epochs=5

# # rlcd run with len balancing
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/rlcdlenbal/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=1 \
#     --num_train_epochs=2

# # rlcd run with lower mix ratio
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/rlcdmix20/ \
#     --dataset="rlcd" \
#     --mix_ratio=0.2 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=2

# # webgpt run with lower mix ratio
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/webgptmix20/ \
#     --dataset="wgpt" \
#     --mix_ratio=0.2 \
#     --rand_ratio=0 \
#     --balance_len=0

# # stack run with balancingy
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/stacklenbal/ \
#     --dataset="stack" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=1 \
#     --num_train_epochs=2

# # stack run with lower mix ratio
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/stackmix20/ \
#     --dataset="stack" \
#     --mix_ratio=0.2 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=2


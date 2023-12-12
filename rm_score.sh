# # 
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/rlcdbigrm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/multisamp/generated_rlcdsftmulti.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0

# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/wgptbigrm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/multisamp/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 600 \
#     --shuffle 0

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/sft13brm.jsonl

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/sftdpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/sftdpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/sft7brm.jsonl

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/sft13brm.jsonl

# export CUDA_VISIBLE_DEVICES=0,1
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u25_dpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=0,1
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u25_dpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u50_dpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u50_dpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u75_dpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u75_dpo13b.jsonl


# export CUDA_VISIBLE_DEVICES=2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/sft_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u25_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u50_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=0,1
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u75_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=0,1
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra100.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u100_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u25_tulu13b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u50_tulu13b.jsonl

# export CUDA_VISIBLE_DEVICES=6,7
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
#     --device 0 \
#     --lim 200 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u75_tulu13b.jsonl

# export CUDA_VISIBLE_DEVICES=6,7
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u200_dpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=6,7
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u400_dpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=6,7
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_dpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=6,7
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u975_dpo13b.jsonl


# export CUDA_VISIBLE_DEVICES=0,1
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u200_tulu13b.jsonl

# export CUDA_VISIBLE_DEVICES=2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u400_tulu13b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_tulu13b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_dpo13b.jsonl

export CUDA_VISIBLE_DEVICES=6,7
python -u scripts/rmsco_outs.py \
    --rmname="allenai/tulu-2-dpo-13b" \
    --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
    --device 0 \
    --lim 100 \
    --shuffle 0 \
    --basemodel "nobase" \
    --outputdir outputs/ultrarmscos/u975_dpo13b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u200_dpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u400_dpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_dpo7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-dpo-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u975_dpo7b.jsonl


# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u200_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u400_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=4,5
# python -u scripts/rmsco_outs.py \
#     --rmname="allenai/tulu-2-7b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u975_tulu7b.jsonl

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u200_13brm.jsonl


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u400_13brm.jsonl


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_13brm.jsonl


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u975_13brm.jsonl

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u200_7brm.jsonl


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u400_7brm.jsonl


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u750_7brm.jsonl


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
#     --device 0 \
#     --lim 100 \
#     --shuffle 0 \
#     --basemodel "nobase" \
#     --outputdir outputs/ultrarmscos/u975_7brm.jsonl

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nnodes 1  --nproc_per_node 1 --master_port=12349 scripts/dpo_eval.py \
#         --model_name=allenai/tulu-2-70b \
#         --output_dir="" \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1 \
#         --per_device_train_batch_size=1 \
#         --per_device_eval_batch_size=1
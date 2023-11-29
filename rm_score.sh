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

export CUDA_VISIBLE_DEVICES=6,7
python -u scripts/rmsco_outs.py \
    --rmname="allenai/tulu-2-dpo-7b" \
    --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra100.jsonl" \
    --device 0 \
    --lim 200 \
    --shuffle 0 \
    --basemodel "nobase" \
    --outputdir outputs/ultrarmscos/u100_dpo7b.jsonl

export CUDA_VISIBLE_DEVICES=6,7
python -u scripts/rmsco_outs.py \
    --rmname="allenai/tulu-2-dpo-13b" \
    --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra100.jsonl" \
    --device 0 \
    --lim 200 \
    --shuffle 0 \
    --basemodel "nobase" \
    --outputdir outputs/ultrarmscos/u100_dpo13b.jsonl
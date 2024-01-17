#!/bin/bash

# Define the common variables
CUDA_DEVICE=0
BASEMODEL="EleutherAI/gpt-neo-125m"
CKNUM=6000
CHECKPOINT_PATH="/u/prasanns/research/rlhf-length-biases/checkpoints/rmunpairmix/checkpoint-${CKNUM}"
DEVICE=0
LIMIT=600
SHUFFLE=0
MAXLEN=50

# Export CUDA visible devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Define a function to run the script with different inputs
run_script() {
    INPUT_FILE="outputs/calibcheck/dpoplusbow${1}.jsonl"
    OUTPUT_DIR="outputs/calibscos/tokenpredbow${1}_rm${CKNUM}"

    python -u scripts/rmsco_outs.py \
        --rmname="$CHECKPOINT_PATH" \
        --basemodel="$BASEMODEL" \
        --inpf="$INPUT_FILE" \
        --device $DEVICE \
        --lim $LIMIT \
        --shuffle $SHUFFLE \
        --outputdir="$OUTPUT_DIR" \
        --maxlen=$MAXLEN \
        --tokenwise=True
}

# Call the function with different arguments
run_script 25
run_script 50
run_script 100
run_script 200


# export CUDA_VISIBLE_DEVICES=2
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow25.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse25_rm50" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow50.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse50_rm50" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow100.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse100_rm50" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow200.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse200_rm50" \
# #     --maxlen=50

# # export CUDA_VISIBLE_DEVICES=2
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/bowrmself25" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow25.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse25_brms25" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/bowrmself25" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow50.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse50_brms25" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/bowrmself25" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow100.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse100_brms25" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/bowrmself25" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow200.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse200_brms25" \
# #     --maxlen=50

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebowrmv2/checkpoint-13000" \
#     --basemodel="facebook/opt-350m" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/revbow/revbowppogold25.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0 \
#     --outputdir="outputs/calibscos/reverse25_rm13k" \
#     --maxlen=50

# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebowrmv2/checkpoint-13000" \
#     --basemodel="facebook/opt-350m" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/revbow/revbowppogold50.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0 \
#     --outputdir="outputs/calibscos/reverse50_rm13k" \
#     --maxlen=50

# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebowrmv2/checkpoint-13000" \
#     --basemodel="facebook/opt-350m" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/revbow/revbowppogold100.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0 \
#     --outputdir="outputs/calibscos/reverse100_rm13k" \
#     --maxlen=50

# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebowrmv2/checkpoint-13000" \
#     --basemodel="facebook/opt-350m" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/revbow/revbowppogold200.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0 \
#     --outputdir="outputs/calibscos/reverse200_rm13k" \
#     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow400.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/reverse400_rm50" \
# #     --maxlen=50


# # ## FOR NOUNS NOW

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/mininounrms" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusnoun25.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/dponoun25_rm" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/mininounrms" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusnoun50.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/dponoun50_rm" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/mininounrms" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusnoun75.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/dponoun75_rm" \
# #     --maxlen=50

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/mininounrms" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusnoun100.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0 \
# #     --outputdir="outputs/calibscos/dponoun100_rm" \
# #     --maxlen=50



# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow50.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow100.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow200.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/calibcheck/dpoplusbow400.jsonl" \
# #     --device 0 \
# #     --lim 600 \
# #     --shuffle 0


# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/wgptbigrm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/multisamp/generated_wgptmultisampset.jsonl" \
# #     --device 1 \
# #     --lim 600 \
# #     --shuffle 0

# # export CUDA_VISIBLE_DEVICES=0
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/sft13brm.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/sftdpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/sftdpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/sft7brm.jsonl

# # export CUDA_VISIBLE_DEVICES=0
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/sft13brm.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u25_dpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u25_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u50_dpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u50_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u75_dpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u75_dpo13b.jsonl


# # export CUDA_VISIBLE_DEVICES=2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultraorig.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/sft_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u25_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u50_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u75_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra100.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u100_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra25.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u25_tulu13b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra50.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u50_tulu13b.jsonl

# # export CUDA_VISIBLE_DEVICES=6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra75.jsonl" \
# #     --device 0 \
# #     --lim 200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u75_tulu13b.jsonl

# # export CUDA_VISIBLE_DEVICES=6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u200_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u400_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u975_dpo13b.jsonl


# # export CUDA_VISIBLE_DEVICES=0,1
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u200_tulu13b.jsonl

# # export CUDA_VISIBLE_DEVICES=2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u400_tulu13b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_tulu13b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u975_dpo13b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u200_dpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u400_dpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_dpo7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-dpo-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u975_dpo7b.jsonl


# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u200_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u400_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=4,5
# # python -u scripts/rmsco_outs.py \
# #     --rmname="allenai/tulu-2-7b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u975_tulu7b.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u200_13brm.jsonl


# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u400_13brm.jsonl


# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_13brm.jsonl


# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra975.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u975_13brm.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra200.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u200_7brm.jsonl


# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra400.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u400_7brm.jsonl


# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/ultragen/generated_ultra750.jsonl" \
# #     --device 0 \
# #     --lim 100 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/ultrarmscos/u750_7brm.jsonl


# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big25.jsonl" \
# #     --device 0 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step25.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big50.jsonl" \
# #     --device 0 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step50.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big75.jsonl" \
# #     --device 1 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step75.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_bigorig.jsonl" \
# #     --device 1 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/sft.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big100.jsonl" \
# #     --device 2 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step100.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big125.jsonl" \
# #     --device 3 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step125.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big150.jsonl" \
# #     --device 4 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step150.jsonl

# # python -u scripts/rmsco_outs.py \
# #     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/ultrarm" \
# #     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/longppo_samples/generated_longppo_big175.jsonl" \
# #     --device 5 \
# #     --lim 1200 \
# #     --shuffle 0 \
# #     --basemodel "nobase" \
# #     --outputdir outputs/longjobscos/step175.jsonl

# # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # torchrun --nnodes 1  --nproc_per_node 1 --master_port=12349 scripts/dpo_eval.py \
# #         --model_name=allenai/tulu-2-70b \
# #         --output_dir="" \
# #         --dataset="/u/prasanns/research/rlhf-length-biases/data/adv_data/" \
# #         --rand_ratio=0 \
# #         --balance_len=0 \
# #         --num_train_epochs=1 \
# #         --per_device_train_batch_size=1 \
# #         --per_device_eval_batch_size=1
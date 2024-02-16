
BOTTOM=0
TOP=200
MLEN=50
BSIZE=1

# Define a function to run the script with different inputs
run_script() {
    # NOTE that we need to feed things in a specific format
    CKPT_FILE="checkpoints/${1}/${2}${3}${4}"
    OUTPUT_DIR="/u/prasanns/research/rlhf-length-biases/outputs/${1}/genouts/${2}${4}"
    
    python -u scripts/generate_outs.py \
        --basemodel="$BASEMODEL" \
        --dset="$DSET" \
        --ckname="$CKPT_FILE" \
        --fname="$OUTPUT_DIR" \
        --bottom=$BOTTOM --top=$TOP  \
        --bsize=$BSIZE \
        --maxlen=$MLEN

    python -u scripts/evalgold.py  --fname="${OUTPUT_DIR}.jsonl" --gfunct="${1}"
}

export CUDA_VISIBLE_DEVICES=6
BASEMODEL="facebook/opt-125m"
DSET="data/ultra/ultrafeeddiff"
TOP=500
BSIZE=4
# run_script "bagofwords" "dpoplusbow50rm" "/step_" 100

export CUDA_VISIBLE_DEVICES=7
# run_script "bagofwords" "dpoplusbow50rm" "/step_" 200

TOP=200
BSIZE=1
DSET="data/contrastivedistill/wikionpolicyprompts"
run_script "contrastivedistill" "contoptprefs_ppo_v2" "/step_" 50
run_script "contrastivedistill" "contoptprefs_ppo_v2" "/step_" 100
run_script "contrastivedistill" "contoptprefs_ppo_v2" "/step_" 200



# Call the function with different arguments
# run_script  "50rmdpoplus100.jsonl" "50rmdpoplus100"
# run_script  "50rmdpoplus200.jsonl" "50rmdpoplus200"
# run_script  "50rmdpoplus300.jsonl" "50rmdpoplus300"
# run_script  "50rmdpoplus450.jsonl" "50rmdpoplus450"

# run_script "contrastivedistill" "samp60k_ppo" "/step_" 50
# run_script "contrastivedistill" "samp60k_ppo" "/step_" 100
# run_script "contrastivedistill" "samp60k_ppo" "/step_" 200
# run_script "contrastivedistill" "samp60k_ppo" "/step_" 400
# TODO do something to deal with base model disparity

# DSET="data/math/mathppoinps"
# BASEMODEL="models/rewards/math/mathsft1300"
# MLEN=100
# run_script "math" "math_ppo_withrm" "/step_" 350
# run_script "math" "math_ppo_withrm" "/step_" 100
# run_script "math" "math_ppo_withrm" "/step_" 200
# run_script "math" "math_ppo_withrm" "/step_" 300

# MLEN=50
# DSET="data/contrastivedistill/wikionpolicyprompts"
# BASEMODEL="facebook/opt-125m"
# run_script "contrastivedistill" "functioncontrastivedistill_offppo" "/step_" 100
# run_script "contrastivedistill" "functioncontrastivedistill_offppo" "/step_" 200
# run_script "contrastivedistill" "functioncontrastivedistill_offppo" "/step_" 400
# run_script "contrastivedistill" "functioncontrastivedistill_offppo" "/step_" 1000

# DSET="data/ultra/ultrafeeddiff"
# run_script "reversebow" "functionreversebow_offppo" "/step_" 100
# run_script "reversebow" "functionreversebow_offppo" "/step_" 200
# run_script "reversebow" "functionreversebow_offppo" "/step_" 400
# run_script "reversebow" "functionreversebow_offppo" "/step_" 1000

# run_script "contrastivedistill" "samp60k_dpo" "/checkpoint-" 1000
# run_script "contrastivedistill" "samp60k_dpo" "/checkpoint-" 2000
# run_script "contrastivedistill" "samp60k_dpo" "/checkpoint-" 4000

# run_script "contrastivedistill" "truncdata60k_dpo" "/checkpoint-" 1000
# run_script "contrastivedistill" "truncdata60k_dpo" "/checkpoint-" 2000
# run_script "contrastivedistill" "truncdata60k_dpo" "/checkpoint-" 4000

# run_script "reversebow" "50kdponpen" "/checkpoint-" 4000
# run_script "reversebow" "truncdponpen" "/checkpoint-" 1000
# run_script "reversebow" "truncdponpen" "/checkpoint-" 4000

# run_script "reversebow" "revppotruncrmnp" "/step_" 50
# run_script "reversebow" "revppotruncrmnp" "/step_" 100
# run_script "reversebow" "revppo50krmnp" "/step_" 50
# run_script "reversebow" "revppo50krmnp" "/step_" 100

# run_script "reversebow" "revppotruncrm" "/revrmtruncppostep_" 100
# run_script "reversebow" "revppotruncrm" "/revrmtruncppostep_" 200
# run_script "reversebow" "revppotruncrm" "/revrmtruncppostep_" 400
# run_script "reversebow" "revppotruncrm" "/revrmtruncppostep_" 600




# python -u scripts/generate_outs.py \
#         --basemodel="$BASEMODEL" \
#         --dset="$DSET" \
#         --ckname="orig" \
#         --fname="outputs/reversebow/opt125ultrasft" \
#         --bottom=$BOTTOM --top=$TOP  \
#         --bsize=1 \
#         --maxlen=$MLEN
    
# python -u scripts/evalgold.py  --fname="outputs/reversebow/opt125ultrasft.jsonl" --gfunct="reversebow"

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     webgpt \
#     "dpo/dpowgpt/checkpoint-4000" \
#     "wgptdpo" \
#     0 500  \
#     1

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     rlcd \
#     "dpo/dporlcd/checkpoint-4000" \
#     "rlcddpo" \
#     0 500  \
#     1

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     ultra \
#     "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_200" \
#     "ultr200" \
#     0 200  \
#     4

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     ultra \
#     "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_400" \
#     "ultra400" \
#     0 200  \
#     4

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="models/sft10k" \
#     --dset=ultra \
#     --ckname="/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_100" \
#     --fname="longppo_big100" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist=

# export CUDA_VISIBLE_DEVICES=4
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/bowoffpolicysft/step_" \
#     --fname="offppo_bow" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="275,100"


# export CUDA_VISIBLE_DEVICES=7
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/nounoffpolicy/step_" \
#     --fname="nounoffpolicy" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="100,200"

# export CUDA_VISIBLE_DEVICES=2
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/readinggrade_ipo/checkpoint-" \
#     --fname="readingipo" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,3000,5000,10000"


# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/tokdense_ipo/checkpoint-" \
#     --fname="tokdenseipo" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,3000,5000,10000"

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/dpoplusbowexpanded/step_" \
#     --fname="outputs/calibcheck/dpoplusbow" \
#     --bottom=0 --top=100  \
#     --bsize=4 \
#     --cklist="50,100"

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/ratiobow/30/checkpoint-" \
#     --fname="outputs/calibcheck/ipo30test" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,2000,4000,5000" \
#     --maxlen=50

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/ratiobow/50/checkpoint-" \
#     --fname="outputs/calibcheck/ipo50test" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,2000,4000,5000" \
#     --maxlen=50

# export CUDA_VISIBLE_DEVICES=2
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/ratiobow/80/checkpoint-" \
#     --fname="outputs/calibcheck/ipo80test" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,2000,4000,5000" \
#     --maxlen=50

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/ratiobow/85/checkpoint-" \
#     --fname="outputs/calibcheck/ipo85test" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,2000,4000,5000" \
#     --maxlen=50

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/ratiobow/50/checkpoint-" \
#     --fname="outputs/calibcheck/ipo50test" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,1000,2000,4000,5000" \
#     --maxlen=50

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/bowdpopluslearnedrm80/step_" \
#     --fname="outputs/bowdporm" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="100,200,300,400" \
#     --maxlen=50

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/ratiobow/ipobow100k/checkpoint-" \
#     --fname="outputs/bowscale/100data_" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --maxlen=50 \
#     --cklist="500,5000,10000,15000"

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/noundpopluslearnedrm/step_" \
#     --fname="outputs/noundpoplusrm" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --maxlen=50 \
#     --cklist="25,50,75,100,125"

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/unnaturalbow/destrung/checkpoint-" \
#     --fname="outputs/unnat/destrung" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --maxlen=50 \
#     --cklist="250,3000,6000,9000"

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=/u/prasanns/research/rlhf-length-biases/data/bowunnatural/noqtest \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/unnaturalbow/noqdestrung/checkpoint-" \
#     --fname="outputs/unnat/noqdestrung" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --maxlen=50 \
#     --cklist="250,3000,6000,9000"

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/readinggradeppo/step_" \
#     --fname="outputs/reading/readingppogold" \
#     --bottom=0 --top=100  \
#     --bsize=6 \
#     --maxlen=50 \
#     --cklist="25,50,200,900"

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     --basemodel="allenai/tulu-2-7b" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/ultra/dpoplus50rm/ultradpoplus50rmstep_" \
#     --fname="outputs/ultrageneralization/50rmdpoplus" \
#     --bottom=0 --top=200  \
#     --bsize=1 \
#     --maxlen=256 \
#     --cklist="100,200,300,450"

# export CUDA_VISIBLE_DEVICES=2
# python -u scripts/generate_outs.py \
#     --basemodel="allenai/tulu-2-7b" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/ultra/dpoplussmalldrm/ultradpoplussmalldiststep_" \
#     --fname="outputs/ultrageneralization/smalldistrrmdpoplus" \
#     --bottom=0 --top=200  \
#     --bsize=1 \
#     --maxlen=256 \
#     --cklist="100,200,300,450"

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/trainablebowactiveconfv2/step_" \
#     --fname="outputs/bowactive/bowactivev1" \
#     --bottom=0 --top=200  \
#     --bsize=1 \


#     --maxlen=50 \
#     --cklist="700,900,1000,1400"

# export CUDA_VISIBLE_DEVICES=4
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="checkpoints/ultra/" \
#     --fname="outputs/ultrageneralization/" \
#     --bottom=0 --top=200  \
#     --bsize=1 \
#     --maxlen=256 \
#     --cklist="100,200"


# export CUDA_VISIBLE_DEVICES=2
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/nounsdpoplus/step_" \
#     --fname="outputs/calibcheck/dpoplusnoun" \
#     --bottom=0 --top=100  \
#     --bsize=4 \
#     --cklist="25,50"

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="/u/prasanns/research/rlhf-length-biases/checkpoints/nounsdpoplus/step_" \
#     --fname="outputs/calibcheck/dpoplusnoun" \
#     --bottom=0 --top=100  \
#     --bsize=4 \
#     --cklist="75,100" \
#     --max_len=50

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     ultra \
#     "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_125" \
#     "longppo_big125v2" \
#     8400 9600  \
#     4

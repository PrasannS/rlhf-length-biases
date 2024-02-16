#!/bin/bash

# Define the common variables

# Export CUDA visible devices
DEVICE=5
export CUDA_VISIBLE_DEVICES=$DEVICE
IND=$DEVICE
# START=$(($IND*20000))
# END=$(($IND+1))
# END=$(($END*20000))

# Define a function to run the script with different inputs
run_script() {
    INPUT_FILE="/u/prasanns/research/rlhf-length-biases/outputs/ultrageneralization/${1}"
    OUTPUT_DIR="/u/prasanns/research/rlhf-length-biases/outputs/ultragenerscos/${2}"

    python -u scripts/rmsco_outs.py \
        --rmname="models/rewards/ultrarm" \
        --inpf="$INPUT_FILE" \
        --device 0 \
        --outputdir="$OUTPUT_DIR"
}

# Call the function with different arguments
# run_script  "50rmdpoplus100.jsonl" "50rmdpoplus100"
# run_script  "50rmdpoplus200.jsonl" "50rmdpoplus200"
# run_script  "50rmdpoplus300.jsonl" "50rmdpoplus300"
# run_script  "50rmdpoplus450.jsonl" "50rmdpoplus450"

run_script  "sfttulu.jsonl" "sfttulu"
# run_script  "ultra13dpoplus200.jsonl" "ultra13dpoplus200"
# run_script  "dpo44tulu3000.jsonl" "dpo44tulu3000"
# run_script  "dpo44tulu4000.jsonl" "dpo44tulu4000"

# run_script  "smalldistdpo1000.jsonl" "smalldistdpo1000"
# run_script  "smalldistdpo2000.jsonl" "smalldistdpo2000"
# run_script  "smalldistdpo4000.jsonl" "smalldistdpo4000"

# run_script  "smalldistrrmdpoplus100.jsonl" "smalldistrrmdpoplus100"
# run_script  "smalldistrrmdpoplus200.jsonl" "smalldistrrmdpoplus200"
# run_script  "smalldistrrmdpoplus300.jsonl" "smalldistrrmdpoplus300"
# run_script  "smalldistrrmdpoplus450.jsonl" "smalldistrrmdpoplus450"





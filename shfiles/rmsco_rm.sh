#!/bin/bash

# Define the common variables

# Export CUDA visible devices
DEVICE=7
export CUDA_VISIBLE_DEVICES=$DEVICE
IND=$DEVICE
START=$(($IND*20000))
END=$(($IND+1))
END=$(($END*20000))

# Define a function to run the script with different inputs
run_script() {
    INPUT_FILE="data/ufdiff_short"
    OUTPUT_DIR="data/${2}"

    python -u scripts/rmsco_outs.py \
        --rmname="${1}" \
        --inpf="$INPUT_FILE" \
        --device 0 \
        --outputdir="$OUTPUT_DIR" \
        --isdset=True \
        --start=$START \
        --lim=$END
}

# Call the function with different arguments
run_script  "models/rewards/ultrarm" "7bdata/{$IND}"

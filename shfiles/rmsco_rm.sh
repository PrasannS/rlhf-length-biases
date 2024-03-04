#!/bin/bash

# Define the common variables

# Export CUDA visible devices
DEVICE=3
export CUDA_VISIBLE_DEVICES=$DEVICE
IND=$DEVICE
# START=$(($IND*20000))
# END=$(($IND+1))
# END=$(($END*20000))

# Define a function to run the script with different inputs
run_script() {
    INPUT_FILE="/u/prasanns/research/rlhf-length-biases/outputs/${1}/${2}.jsonl"
    OUTPUT_DIR="/u/prasanns/research/rlhf-length-biases/outputs/${1}/${2}_${3}_sco"

    python -u scripts/rmsco_outs.py \
        --rmname="models/rewards/${1}/${3}" \
        --inpf="$INPUT_FILE" \
        --device 0 \
        --outputdir="$OUTPUT_DIR" \
        --maxlen=50 \
        --gfunct=${1}
}

run_script_dset() {
    INPUT_FILE="data/${1}/${2}"

    python -u scripts/rmsco_outs.py \
        --rmname="models/rewards/${1}/${3}" \
        --inpf="$INPUT_FILE" \
        --device 0 \
        --maxlen=-1 \
        --start=0 \
        --lim=200 \
        --isdset=True
}

# run_script_dset bagofwords "nozero100k" "expbow50"
# run_script_dset bagofwords "nozero100k" "nozero100k_125mag_rm"
# run_script_dset bagofwords "nozero100k" "nozero100k_125magnofa_rm"
# run_script_dset bagofwords "nozero100k" "nozero100k_125popnofa_rm"

run_script_dset bagofwords "nozero100k" "nozero100k_125runagainyesfa_rm"

# run_script bagofwords "calibcheck/dpoplusbow400" "nozero100k_125popnofa_rm"
# run_script bagofwords "calibcheck/dpoplusbow200" "nozero100k_125popnofa_rm"
# run_script bagofwords "calibcheck/dpoplusbow100" "nozero100k_125popnofa_rm"
# run_script bagofwords "calibcheck/dpoplusbow25" "nozero100k_125popnofa_rm"

# run_script bagofwords "calibcheck/dpoplusbow400" "nozero100k_125magnofa_rm"
# run_script bagofwords "calibcheck/dpoplusbow200" "nozero100k_125magnofa_rm"
# run_script bagofwords "calibcheck/dpoplusbow100" "nozero100k_125magnofa_rm"
# run_script bagofwords "calibcheck/dpoplusbow25" "nozero100k_125magnofa_rm"


# run_script bagofwords "calibcheck/dpoplusbow400" "nozero100k_350mag_rm"
# run_script bagofwords "calibcheck/dpoplusbow200" "nozero100k_350mag_rm"
# run_script bagofwords "calibcheck/dpoplusbow100" "nozero100k_350mag_rm"

# /u/prasanns/research/rlhf-length-biases/models/rewards/bagofwords/nozero100k_125mag_rm

# run_script bagofwords "calibcheck/dpoplusbow25" "nozero100k_125renorm_rm"
# run_script bagofwords "calibcheck/dpoplusbow100" "nozero100k_125renorm_rm"
# run_script bagofwords "calibcheck/dpoplusbow200" "nozero100k_125renorm_rm"
# run_script bagofwords "calibcheck/dpoplusbow400" "nozero100k_125renorm_rm"

run_script nouns "calibcheck/dpoplusnoun25" "dponounsynth_125magnfa_rm"
run_script nouns "calibcheck/dpoplusnoun50" "dponounsynth_125magnfa_rm"
run_script nouns "calibcheck/dpoplusnoun75" "dponounsynth_125magnfa_rm"
run_script nouns "calibcheck/dpoplusnoun100" "dponounsynth_125magnfa_rm"


run_script nouns "calibcheck/dpoplusnoun25" "dponounsynth_125poverpnfa_rm"
run_script nouns "calibcheck/dpoplusnoun50" "dponounsynth_125poverpnfa_rm"
run_script nouns "calibcheck/dpoplusnoun75" "dponounsynth_125poverpnfa_rm"
run_script nouns "calibcheck/dpoplusnoun100" "dponounsynth_125poverpnfa_rm"

# Call the function with different arguments
# run_script  "50rmdpoplus100.jsonl" "50rmdpoplus100"
# run_script  "50rmdpoplus200.jsonl" "50rmdpoplus200"
# run_script  "50rmdpoplus300.jsonl" "50rmdpoplus300"
# run_script  "50rmdpoplus450.jsonl" "50rmdpoplus450"

# run_script  "sfttulu.jsonl" "sfttulu"
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





# NOTE that this endpoint needs to match with the desired reward model 
# TODO maybe specify an extra port somehow

SUPDATES=1000000
noupdateapi() {
    # NOTE that we need to feed things in a specific format    

    # TODO model_name might not mean anything
    python -u rmapi/rmapi.py \
        --model_name=facebook/opt-125m \
        --dataset_name="data/${1}/${2}" \
        --reward_model_name=models/rewards/${1}/${3} \
        --save_freq=50 \
        --max_length=256 --batch_size=16 \
        --seed=0 --learning_rate=5e-5 \
        --trainable --goldreward="function${1}" \
        --stopupdates=$SUPDATES \
        --output_dir="checkpoints/${1}/dynarm_${4}/" \
        --logfile="outputs/${1}/dynarmlogs/${3}_${4}.jsonl" \
        --port=${5} \
        --tracking
}

# SUPDATES=3200
# # export CUDA_VISIBLE_DEVICES=4
# # noupdateapi "contrastivedistill" "contdfixed" "contoptprefs_rm" "_log" 5000
# export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth100k" "nozero100k_125runagainnofa_rm" "50steps" 5001

# SUPDATES=12800
# # export CUDA_VISIBLE_DEVICES=4
# # noupdateapi "contrastivedistill" "contdfixed" "contoptprefs_rm" "_log" 5000
# export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth100k" "nozero100k_125runagainnofa_rm" "200steps" 5003

# SUPDATES=6400
# # export CUDA_VISIBLE_DEVICES=4
# # noupdateapi "contrastivedistill" "contdfixed" "contoptprefs_rm" "_log" 5000
# export CUDA_VISIBLE_DEVICES=1
# noupdateapi "bagofwords" "bowsynth100k" "nozero100k_125runagainnofa_rm" "100steps" 5002

SUPDATES=10000000
# export CUDA_VISIBLE_DEVICES=4
# noupdateapi "contrastivedistill" "contdfixed" "contoptprefs_rm" "_log" 5000
export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "updatereprod2" 5000
export CUDA_VISIBLE_DEVICES=0
noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "updatereprod_nobase3" 5000
# SUPDATES=1600
# # export CUDA_VISIBLE_DEVICES=4
# # noupdateapi "contrastivedistill" "contdfixed" "contoptprefs_rm" "_log" 5000
# export CUDA_VISIBLE_DEVICES=4
# noupdateapi "bagofwords" "mathprefdata" "math_rm" "log" 5001
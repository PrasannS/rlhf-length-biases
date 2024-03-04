export CUDA_VISIBLE_DEVICES=4,5

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=1238 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/contrastivedistill/contdistillrmnormal \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/contrastivedistill/contrastdistillprefs" \
#         --rand_ratio=0 \
#         --evaldata="/u/prasanns/research/rlhf-length-biases/data/contrastivedistill/contrastdistillheldoutprefs" \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=32 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --eval_steps=250 \
#         --learning_rate=5e-5

BSIZE=32
LTYPE="normal"
NOLORA=True
REINIT=True
run_script() {
    echo $LR
    echo $BASEMODEL
    torchrun --nnodes 1  --nproc_per_node 2 --master_port=${4} scripts/train_rm.py \
        --model_name=$BASEMODEL"" \
        --output_dir=checkpoints/${1}/${2}_${5}_rm/ \
        --dataset="data/${1}/${2}" \
        --rand_ratio=0 \
        --evaldata="data/${1}/${3}" \
        --balance_len=0 \
        --num_train_epochs=2 \
        --per_device_train_batch_size=$BSIZE \
        --per_device_eval_batch_size=8 \
        --gradient_accumulation_steps=1 \
        --eval_steps=100 \
        --save_steps=1000 \
        --learning_rate=$LR \
        --losstype=$LTYPE \
        --nolora=$NOLORA \
        --random_reinit=$REINIT

    python scripts/merge_peft_adapter.py \
        --adapter_model_name="checkpoints/${1}/${2}_${5}_rm/best_checkpoint" \
        --base_model_name="$BASEMODEL" \
        --output_name="models/rewards/${1}/${2}_${5}_rm"
}

# BASEMODEL="/u/prasanns/research/rlhf-length-biases/models/rewards/math/mathsft1300"
# run_script "bagofwords" "bowprefseqlenprefs" "bowsynth100k" 12350
BASEMODEL="facebook/opt-1.3b"
# BASEMODEL="facebook/opt-125m"
# BASEMODEL="facebook/opt-125m"
# BASEMODEL="facebook/opt-350m"

# BASEMODEL="meta-llama/Llama-2-7b-hf"
# export CUDA_VISIBLE_DEVICES=0,1
# run_script "contrastivedistill" "contoptprefs" "contoptprefs" 12349 "1b"
BSIZE=4
LR=5e-5
NOLORA=True
export CUDA_VISIBLE_DEVICES=0,1
run_script "math" "mathprefdata" "mathprefdata" 12321 "1brandinitnolorablr"

# LR=1e-4
# export CUDA_VISIBLE_DEVICES=2,3
# run_script "math" "mathprefdata" "mathprefdata" 12352 "125v4"


# BSIZE=4
# export CUDA_VISIBLE_DEVICES=4,5
# LR=7e-5
# LTYPE="mag"
# run_script "nouns" "dponounsynth" "dponounsynth" 12311 "1bnounrmnofa"
# LTYPE="prefoverpref"
# run_script "bagofwords" "nozero100k" "nozero100k" 12349 "125popnofa"

# LTYPE="mag"
# run_script "nouns" "dponounsynth" "dponounsynth" 12349 "125magnfa"
# LTYPE="prefoverpref"
# run_script "nouns" "dponounsynth" "dponounsynth" 12349 "125poverpnfa"

LR=6e-5
# LTYPE="mag"
# run_script "math" "mathprefdata" "mathprefdata" 12349 "125magnfa"
# LTYPE="prefoverpref"
# run_script "math" "mathprefdata" "mathprefdata" 12349 "125prefoprefnfa"
# run_script "math" "mathprefdata" "mathprefdata" 12348 "125mnfa"
# BASEMODEL="facebook/opt-1.3b"
# BSIZE=4
# run_script "math" "mathprefdata" "mathprefdata" 12348 "1bmnfa"

# export CUDA_VISIBLE_DEVICES=4,5
# BASEMODEL="facebook/opt-1.3b"
LR=3e-5
# run_script "bagofwords" "nozero100k" "nozero100k" 12353 "350mag"
# run_script "bagofwords" "nozero100k" "nozero100k" 12350 "350norm"
# export CUDA_VISIBLE_DEVICES=4,5
# run_script "math" "mathprefdata" "mathprefdata" 12350 "1bmag"
# LTYPE="prefoverpref"
# run_script "math" "mathprefdata" "mathprefdata" 12351 "125prefopref"

# BSIZE=12

# export CUDA_VISIBLE_DEVICES=6,7
# run_script "nouns" "dponounsynth" "dponounsynth" 12352 "125mag"
# LTYPE="prefoverpref"
# run_script "nouns" "dponounsynth" "dponounsynth" 12352 "125poverp"

# NOTE merge math with the right SFT model

# run_script "contrastivedistill" "truncdata60k" "heldoutprefs" 12349


# torchrun --nnodes 1  --nproc_per_node 2 --master_port=1239 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/reversebow/truncrmnp/ \
#         --dataset="data/contrastivedistill/samp60k" \
#         --rand_ratio=0 \
#         --evaldata="/u/prasanns/research/rlhf-length-biases/data/revbow/revbowtest" \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=32 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --eval_steps=250 \
#         --learning_rate=3e-5


# torchrun --nnodes 1  --nproc_per_node 2 --master_port=1239 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=checkpoints/reversebow/50krmnp/ \
#         --dataset="data/contrastivedistill/truncdata60k" \
#         --rand_ratio=0 \
#         --evaldata="/u/prasanns/research/rlhf-length-biases/data/revbow/revbowtest" \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=32 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --eval_steps=250 \
#         --learning_rate=3e-5

# Do cross eval
# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12337 scripts/stack_cross_eval.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
#         --output_dir=checkpoints/stackrms/stack_workplace \
#         --dataset="stack_workplace" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12337 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=checkpoints/ultranormal \
#         --dataset="ultra" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10mag7b \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2

# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12339 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/rlcdbighyperexplore/ \
#         --dataset="rlcd" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2 \
#         --per_device_train_batch_size=1 \
#         --per_device_eval_batch_size=1 \
#         --gradient_accumulation_steps=8 \
#         --learning_rate=1e-4

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# torchrun --nnodes 1  --nproc_per_node 6 --master_port=12339 scripts/train_rm.py \
#         --model_name=allenai/tulu-2-70b \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/ultrafeed/giganticrm \
#         --dataset="ultra" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=30 \
#         --per_device_train_batch_size=1 \
#         --per_device_eval_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --learning_rate=5e-5 \
#         --eval_steps=50 \
#         --gradient_checkpointing=true
        #\
        #--deepspeed=/u/prasanns/research/rlhf-length-biases/scripts/deepspeed_config.json

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10truncmag/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun90trunc \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=30 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun90linmag/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun90lin \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10truncmagv2/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun10trunc \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50


# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/treedep/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/treedep \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10v2sanity/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/synthnoun10truncv2 \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/triplenoconflict/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/3synthdata \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12339 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/moresynth/bowrm/ \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/dpobowsynth \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=15 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=50

# export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12341 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/rmself75 \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/rmself/75self \
#         --evaldata=/u/prasanns/research/rlhf-length-biases/data/bowtrunc/heldouttest \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=500


# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12341 scripts/train_rm.py \
#         --model_name=meta-llama/Llama-2-7b-hf \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/ultra50rm \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/ultra50k \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2 \
#         --gradient_accumulation_steps=2 \
#         --learning_rate=1e-5 \
#         --eval_steps=500

# export CUDA_VISIBLE_DEVICES=3,4
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12342 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/distillrm \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/distilprefdata \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=16 \
#         --per_device_eval_batch_size=2 \
#         --gradient_accumulation_steps=2 \
#         --learning_rate=5e-5 \
#         --eval_steps=500

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12342 scripts/train_rm.py \
#         --model_name=models/sftwiki \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/undoprior \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/distilprefdata \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=16 \
#         --per_device_eval_batch_size=2 \
#         --gradient_accumulation_steps=2 \
#         --learning_rate=5e-5 \
#         --eval_steps=500

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12342 scripts/train_rm.py \
#         --model_name=models/sfteinstein \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/einsteinrm2layer \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/einstein2house \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=16 \
#         --per_device_eval_batch_size=2 \
#         --gradient_accumulation_steps=2 \
#         --learning_rate=5e-5 \
#         --eval_steps=500

# export CUDA_VISIBLE_DEVICES=4
# python scripts/train_rm.py \
#         --model_name=EleutherAI/gpt-neo-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/rmunpairmix \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/80 \
#         --evaldata=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=2 \
#         --learning_rate=1e-4 \
#         --eval_steps=500 \
#         --tokenbased=True

# export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12340 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/50rmv2 \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/50 \
#         --evaldata=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=500

# export CUDA_VISIBLE_DEVICES=6,7
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12340 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/70rmv2 \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/70 \
#         --evaldata=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=3 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=500

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12341 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/80rmv2 \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/80 \
#         --evaldata=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=500

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12341 scripts/train_rm.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/85rmv2 \
#         --dataset=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/85 \
#         --evaldata=/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=2 \
#         --per_device_train_batch_size=8 \
#         --per_device_eval_batch_size=32 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=1e-4 \
#         --eval_steps=500


# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12339 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun10mag7b \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/synthnoun10" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=10 \
#         --per_device_train_batch_size=2 \
#         --per_device_eval_batch_size=2
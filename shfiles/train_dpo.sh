# # DPO setup for WebGPT
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/sft10k" --output_dir="dpo/dpoultra50" \
#     --dataset="data/ultra50k" 

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="allenai/tulu-2-7b" --output_dir="dpo/dpoultasmalldistr" \
#     --dataset="data/ultrarmsmall" \
#     --per_device_train_batch_size=1 \
#     --gradient_accumulation_steps=32

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/sfteinstein" --output_dir="dpo/einsteindpo2layers" \
#     --dataset="data/einstein2house" \
#     --per_device_train_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --epochs=3 \
#     --learning_rate=5e-5 \
#     --promptstyle="ans"

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/distillwikipref" \
#     --dataset="data/distilprefdata" \
#     --per_device_train_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --epochs=3 \
#     --learning_rate=5e-5 \
#     --promptstyle="onlyans"
# Hmm this wasn't set low enough
BETA=0.05
run_script() {

    accelerate launch --config_file=scripts/default_single.yaml --main_process_port=${5} \
        scripts/train_dpo.py \
        --model_name_or_path="$BASEMODEL" --output_dir="checkpoints/${1}/${2}_${4}_dpo/" \
        --dataset="data/${1}/${2}" \
        --per_device_train_batch_size=8 \
        --gradient_accumulation_steps=4 \
        --per_device_eval_batch_size=8 \
        --epochs=5 \
        --evaldata="data/${1}/${3}" \
        --learning_rate=3e-5 \
        --beta=$BETA \
        --save_steps=50 \
        --eval_steps=200

}
# export CUDA_VISIBLE_DEVICES=0
# BASEMODEL="models/bagofwords/50rmppo_s100_sft"
# # run_script "contrastivedistill" "contdfixed" "heldoutprefs" 29522
# run_script "bagofwords" "bowmax2" "bowmax2" "s100sft" 29522
# BASEMODEL="models/bagofwords/smalldist_sft"
# BASEMODEL="models/bagofwords/50rmppo_s200_sft"
# run_script "bagofwords" "bowmax2" "bowmax2" "smallsft" 29524
BASEMODEL="facebook/opt-125m"
# export CUDA_VISIBLE_DEVICES=0
# run_script "nouns" "nouns_revlabel" "nouns_revlabel" "revdpo" 29524

# export CUDA_VISIBLE_DEVICES=6
# run_script "bagofwords" "3kprefs" "3kheld" "smalldpo" 29525
# run_script "contrastivedistill" "3kprefs" "3kheld" "smalldpo" 29525

export CUDA_VISIBLE_DEVICES=7
run_script "nouns" "3kprefs" "3kheld" "smalldpo" 29526
run_script "math" "3kprefs" "3kheld" "smalldpo" 29526


# export CUDA_VISIBLE_DEVICES=4
# BETA=0.01
# run_script "bagofwords" "nozero100k" "nozero100k" "betapt01" 29525

# export CUDA_VISIBLE_DEVICES=3
# BETA=1
# run_script "bagofwords" "nozero100k" "nozero100k" "beta1" 29526

# export CUDA_VISIBLE_DEVICES=0
# BETA=0.00001
# # run_script "bagofwords" "nozero100k" "nozero100k" "betatiny" 29527
# export CUDA_VISIBLE_DEVICES=1
# BETA=0.00001
# run_script "nouns" "dponounsynth" "dponounsynth" "betatiny" 29526

# export CUDA_VISIBLE_DEVICES=2
# BASEMODEL="facebook/opt-125m"
# BETA=0.01
# run_script "nouns" "dponounsynth" "dponounsynth" "betapt01nofa" 29527
# export CUDA_VISIBLE_DEVICES=3
# BETA=1
# run_script "nouns" "dponounsynth" "dponounsynth" "betabig" 29528

# export CUDA_VISIBLE_DEVICES=0
# BASEMODEL="models/rewards/math/mathsft1300"
# BETA=0.01
# run_script "math" "mathprefdata" "mathprefdata" "betapt01" 29525
# export CUDA_VISIBLE_DEVICES=1
# BETA=1
# run_script "math" "mathprefdata" "mathprefdata" "betabig" 29526

# BASEMODEL="models/bagofwords/tinybow_sft"
# export CUDA_VISIBLE_DEVICES=2

# run_script "bagofwords" "bowmax2" "bowmax2" "tinysft" 29526
# export CUDA_VISIBLE_DEVICES=3
# run_script "bagofwords" "nozero100k" "nozero100k" "tinysft" 29527

# export BASEMODEL="facebook/opt-125m"
# export CUDA_VISIBLE_DEVICES=3
# run_script "bagofwords" "nozero100k" "nozero100k" "normsft" 29527

# export CUDA_VISIBLE_DEVICES=2
# run_script "bagofwords" "bowmax2" "bowmax2" "normsft" 29526

# BASEMODEL="models/bagofwords/50rmppo_s100_sft"

# export CUDA_VISIBLE_DEVICES=4
# run_script "bagofwords" "nozero100k" "nozero100k" "normsft" 29526


# run_script "contrastivedistill" "samp60k" "heldoutprefs" 29522
# run_script "contrastivedistill" "truncdata60k" "heldoutprefs" 12349


# export CUDA_VISIBLE_DEVICES=7
# accelerate launch --config_file=scripts/default_single.yaml --main_process_port=29522 \
#     scripts/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="checkpoints/reversebow/50kdponpen/" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/revbow/revbow50knopen" \
#     --per_device_train_batch_size=8 \
#     --gradient_accumulation_steps=4 \
#     --epochs=3 \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/revbow/revbowtest" \
#     --learning_rate=3e-5

# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --config_file=scripts/default_single.yaml --main_process_port=29522 \
#     scripts/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="checkpoints/reversebow/truncdponpen/" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/revbow/revbowtruncnopen" \
#     --per_device_train_batch_size=8 \
#     --gradient_accumulation_steps=4 \
#     --epochs=3 \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/revbow/revbowtest" \
#     --learning_rate=3e-5

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29524 \
#     scripts/train_dpo.py \
#     --model_name_or_path="allenai/tulu-2-7b" --output_dir="dpo/dpoultra44v2" \
#     --dataset="data/ultra44k" \
#     --per_device_train_batch_size=1 \
#     --gradient_accumulation_steps=8 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/sftwiki" --output_dir="dpo/undopriorwiki" \
#     --dataset="data/distilprefdata" \
#     --per_device_train_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --epochs=3 \
#     --learning_rate=5e-5 \
#     --promptstyle="onlyans"


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29527  \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/expandedbowsynth_kto" \
#     --dataset="data/expandedbowsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5 \
#     --loss_type=kto_pair \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --eval_steps=200 \
#     --save_steps=250 \
#     --max_steps=10000

# DPO setup for Stack
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/stack/sft" --output_dir="dpo/dpostack" \
#     --dataset="stack"

# # DPO setup for RLCD
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dponoun" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/dponounsynth"

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobow_pposimsort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_sort" \
#     --evaldata="data/dpobowsynth"

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobow_pposimrand" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_random" \
#     --evaldata="data/dpobowsynth"

# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dponoun_pposimsort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/noundpo_pposim_sorted" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/nounsynthadjhyper" \
#     --dataset="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/bowsynthadjhyper" \
#     --dataset="data/dpobowsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobowendonlysort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_endonly" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpobowendonlynosort" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_nosort_endonly" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpopposortshuffled" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_sorted" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpopposortshuffled" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowdpo_pposim_sorted" \
#     --evaldata="data/dponounsynth" \
#     --beta=0.02

# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dpopcsynth" \
#     --dataset="data/poscontextsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/pcsynth_ipo" \
#     --dataset="data/poscontextsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5 \
#     --loss_type=ipo


# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/tokdense_ipo" \
#     --dataset="data/dpouniquetokratio" \
#     --beta=0.02 \
#     --learning_rate=5e-5 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --max_steps=10000

# export CUDA_VISIBLE_DEVICES=7
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/30" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/30" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=7
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/50" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/50" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=7
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/70" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/70" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/ipobow100k" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowsynth100k" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/ipobow250k" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowsynth250k" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --eval_steps=1000 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/unnaturalbow/noquestions" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowunnatural/noquestions" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/bowunnatural/noqtest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --eval_steps=1000 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/unnaturalbow/noqdestrung" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowunnatural/destrungnoq" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/bowunnatural/destrungtestnoq" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --eval_steps=1000 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29527 \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/uncommonbow" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowtrunc/train100k" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/bowtrunc/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --eval_steps=1000 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/ipobow250k" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/bowsynth250k" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --eval_steps=1000 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/ratiobow/85" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/85" \
#     --evaldata="/u/prasanns/research/rlhf-length-biases/data/ratiovarbow/heldouttest" \
#     --beta=0.02 \
#     --learning_rate=1e-4 \
#     --loss_type=ipo \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --save_steps=250 \
#     --epochs=3

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29527  \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/expandedbowsynth_kto" \
#     --dataset="data/expandedbowsynth" \
#     --beta=0.02 \
#     --learning_rate=5e-5 \
#     --loss_type=kto_pair \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --eval_steps=200 \
#     --save_steps=250 \
#     --max_steps=10000

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --multi_gpu --config_file=scripts/default_dpomulti.yaml --main_process_port=29528  \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/expandedbowsynth_dporerun" \
#     --dataset="data/expandedbowsynth" \
#     --beta=0.02 \
#     --learning_rate=1e-6 \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --eval_steps=200 \
#     --save_steps=250 \
#     --max_steps=10000

# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml \
#     --main_process_port=29527 dpo_exps/train_dpo.py \
#     --model_name_or_path="allenai/tulu-2-7b" --output_dir="dpo/ultrarealisticdpofollowppo" \
#     --dataset="data/ultrarealdpo" \
#     --evaldata="data/ultrafeeddiff" \
#     --beta=0.1 \
#     --learning_rate=1e-6 \
#     --gradient_accumulation_steps=4 \
#     --eval_steps=100 \
#     --save_steps=300 \
#     --max_steps=10000

# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="facebook/opt-125m" --output_dir="dpo/dponoun_pposimrand" \
#     --dataset="/u/prasanns/research/rlhf-length-biases/data/noundpo_pposim_random" \
#     --evaldata="data/dponounsynth"
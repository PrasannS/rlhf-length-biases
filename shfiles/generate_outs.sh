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

export CUDA_VISIBLE_DEVICES=0
python -u scripts/generate_outs.py \
    --basemodel="allenai/tulu-2-7b" \
    --dset=ultra \
    --ckname="/u/prasanns/research/rlhf-length-biases/dpo/dpoultra44v2/checkpoint-" \
    --fname="outputs/ultrageneralization/dpo44tulu" \
    --bottom=0 --top=200  \
    --bsize=1 \
    --maxlen=256 \
    --cklist="1000,2000,3000,4000"

# export CUDA_VISIBLE_DEVICES=4
# python -u scripts/generate_outs.py \
#     --basemodel="allenai/tulu-2-dpo-7b" \
#     --dset=ultra \
#     --ckname="orig" \
#     --fname="outputs/ultrageneralization/normaldpo" \
#     --bottom=0 --top=200  \
#     --bsize=1 \
#     --maxlen=256


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

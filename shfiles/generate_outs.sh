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

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/nounsynthadjhyper/checkpoint-1000" \
#     "nounadjdpo1k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/nounsynthadjhyper/checkpoint-20000" \
#     "nounadjdpo20k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/bowsynthadjhyper/checkpoint-20000" \
#     "bowadjdpo20k" \
#     0 100  \
#     1


# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/nounsynthadjhyper/checkpoint-20000" \
#     "nounadjdpo20k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/bowsynthadjhyper/checkpoint-20000" \
#     "bowadjdpo20k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dpobowendonlynosort/checkpoint-1000" \
#     "bownosortdpopp1k" \
#     0 100  \
#     1


# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dpobowendonlynosort/checkpoint-20000" \
#     "bownosortdpopp20k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=4
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dpobowendonlysort/checkpoint-1000" \
#     "bowsortdpopp1k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=2
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset="ultra" \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/expandedbowsynth/checkpoint-" \
#     --fname="outputs/fancydpogen/expdpo" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="1000,10000,20000,30000,40000,49000" \
#     --maxlen=100

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset="data/poscontextsynth" \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/dpopcsynth/checkpoint-" \
#     --fname="outputs/fancydpogen/pcsynthdpo" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="1000,10000,20000,30000,40000,49000" \
#     --maxlen=100

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset=ultra \
#     --ckname="dpo/expandedbowsynth_ipo/checkpoint-" \
#     --fname="outputs/fancydpogen/ipobow" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="250,500,750,8000" \
#     --maxlen=100

export CUDA_VISIBLE_DEVICES=2
python -u scripts/generate_outs.py \
    --basemodel="facebook/opt-125m" \
    --dset=ultra \
    --ckname="dpo/expandedbowsynth_kto/checkpoint-" \
    --fname="outputs/fancydpogen/ktopow" \
    --bottom=0 --top=100  \
    --bsize=1 \
    --cklist="500,1000,2000,8000" \
    --maxlen=100

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     --basemodel="facebook/opt-125m" \
#     --dset="ultra" \
#     --ckname="/u/prasanns/research/rlhf-length-biases/dpo/expandedbowsynth/checkpoint-" \
#     --fname="outputs/fancydpogen/pcsynthdpo" \
#     --bottom=0 --top=100  \
#     --bsize=1 \
#     --cklist="1000,10000,20000,30000,40000,49000" \
#     --maxlen=100

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dpopposortshuffled/checkpoint-1000" \
#     "bowsortshuffdpopp1k" \
#     0 100  \
#     1


# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dpopposortshuffled/checkpoint-20000" \
#     "bowsortshuffdpopp20k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=4
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/bowsynthadjhyper/checkpoint-1000" \
#     "bowadjdpo1k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=4
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/bowsynthadjhyper/checkpoint-10000" \
#     "bowadjdpo10k" \
#     0 100  \
#     1

# export CUDA_VISIBLE_DEVICES=5
# python -u scripts/generate_outs.py \
#     "/u/prasanns/research/rlhf-length-biases/models/sft10k" \
#     ultra \
#     "orig" \
#     "longppo_bigorig" \
#     0 1600  \
#     4

# export CUDA_VISIBLE_DEVICES=5
# python -u scripts/generate_outs.py \
#     "/u/prasanns/research/rlhf-length-biases/models/sft10k" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/checkpoints/ultrappolongrun/step_25" \
#     "longppo_big25" \
#     1600 3200  \
#     4

# export CUDA_VISIBLE_DEVICES=6
# python -u scripts/generate_outs.py \
#     "/u/prasanns/research/rlhf-length-biases/models/sft10k" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/checkpoints/ultrappolongrun/step_50" \
#     "longppo_big50" \
#     3200 4800  \
#     4

# export CUDA_VISIBLE_DEVICES=6
# python -u scripts/generate_outs.py \
#     "/u/prasanns/research/rlhf-length-biases/models/sft10k" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/checkpoints/ultrappolongrun/step_75" \
#     "longppo_big75" \
#     4800 5600  \
#     4

# export CUDA_VISIBLE_DEVICES=7
# python -u scripts/generate_outs.py \
#     "/u/prasanns/research/rlhf-length-biases/models/sft10k" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/checkpoints/ultrappolongrun/step_150" \
#     "longppo_big150" \
#     9600 11200  \
#     4

# export CUDA_VISIBLE_DEVICES=7
# python -u scripts/generate_outs.py \
#     "/u/prasanns/research/rlhf-length-biases/models/sft10k" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/checkpoints/ultrappolongrun/step_175" \
#     "longppo_big175v2" \
#     11472 12800  \
#     4

# export CUDA_VISIBLE_DEVICES=8
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/checkpoints/ultrappolongrun/step_50" \
#     "dposimbowrandv2" \
#     3200 4800  \
#     4
# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dponoun_pposimsort/checkpoint-10000" \
#     "dposimnounsort" \
#     0 100  \
#     4

# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dponoun_pposimrand/checkpoint-10000" \
#     "dposimnounrand" \
#     0 100  \
#     4

    

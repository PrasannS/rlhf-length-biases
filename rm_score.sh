# # 
# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/rlcdbigrm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/multisamp/generated_rlcdsftmulti.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0

# python -u scripts/rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-length-biases/models/rewards/wgptbigrm" \
#     --inpf="/u/prasanns/research/rlhf-length-biases/outputs/multisamp/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 600 \
#     --shuffle 0

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u scripts/rmsco_outs.py \
    --rmname="/home/prasann/Projects/rlhf-length-biases/models/rewards/harmrm" \
    --inpf="/home/prasann/Projects/rlhf-exploration/scripts/genouts/multisamp/generated_rlcdsftmulti.jsonl" \
    --device 0 \
    --lim 600 \
    --shuffle 0




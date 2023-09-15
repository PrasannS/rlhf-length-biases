# adapter for llama webgpt model
#python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/wgptllama/checkpoint-4000" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmodel/"

# adapter for steamSHP llama reward model
#python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/shp-llama/checkpoints/llamashp/checkpoint-19000" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/shp-llama/models/shprm/"

# adapter for SHP llama sft model
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/shp-llama/checkpoints/llamasft/checkpoint-2000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/shp-llama/models/shpsft/"

# adapter for adv DA model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/checkpoints/rlcdsaved/checkpoint-3500" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/helprm7b/"

# # adapter for RLCD reward model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advda/checkpoint-3500" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardadvda/"

export CUDA_VISIBLE_DEVICES=0,1
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advtiebreak/checkpoint-4000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardadvtiebreak/"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advmse/checkpoint-8000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardadvmse/"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/rm_da_randonly/checkpoint-4000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardrandaug/"

# new bigdata model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/rm_bigdata/checkpoint-16000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardbigdset/"

# webgpt with random DA
# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/webgptrandda/checkpoint-1500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardrandda/"

# webgpt with TFRM (first attempt)
# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/tfrtest/checkpoint-4500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardtfr"

# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/webgptmix/checkpoint-1500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/mixrm"

# adapter for adv DA model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/checkpoints/rlcdsaved/checkpoint-2000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdhelp/"

# adapter for adv DA model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/rmmix50/checkpoint-2500" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/mix50rm"

# # APFarm human RM 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/apf/checkpoints/apfinithuman/checkpoint-1000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/humanrm"

# # APFarm GPT4 RM
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/apf/checkpoints/apfgptrm/checkpoint-1500" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/gptrm"

#APFarm GPTNEO SFT 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/alpsft/checkpoint-1000" \
#     --base_model_name="EleutherAI/gpt-neo-1.3B" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/neosft"

# #APFarm Human RM attempt #2
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/apf/checkpoints/apfhumrmlbase/checkpoint-2000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/humannew"

# RLCD New RM 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/checkpoints/wgptsaved/checkpoint-2000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdrm"

# # RLCD DA RM
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/checkpoints/da20rlcd/checkpoint-3000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcddarm"

# # WebGPT Mix RM
# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/rmmix50/checkpoint-6000" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmixwgpt/"

# # WebGPT Len Balance RM
# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/rmlenbalance/checkpoint-3500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/lenbalance/"

# #APFarm GPT RM attempt #2
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/apf/checkpoints/apfgptlbase/checkpoint-4500" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/gptcorrect"

# # RLCD Mix 20
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/rlcdmix20/checkpoint-11000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdmix"

# # RLCD Length balancing
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/rlcdlenbal/checkpoint-7000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdlenbal"

# APFarm new style base model for warm starting 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/alpsftllama/checkpoint-1000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft"

# # APFarm with data carto truncation
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/apftruncbothrm/checkpoint-5000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/rmtruncboth"

# APF with data carto both only
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/apftruncbadrm/checkpoint-8000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/rmtruncbad"

# python merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/rlcdrightonly/checkpoint-6500" \
#     --base_model_name="/u/prasanns/research/rlhf-exploration/models/sft" \
#     --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/rmtruncrightrlcd"

# python merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/rlcdleftonly/checkpoint-11000" \
#     --base_model_name="/u/prasanns/research/rlhf-exploration/models/sft" \
#     --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/rmtruncleftrlcd"

# python merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/rlcddiagboth/checkpoint-9500" \
#     --base_model_name="/u/prasanns/research/rlhf-exploration/models/sft" \
#     --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/rmtruncdiag"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/apfranddav3/checkpoint-20500" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/apfrda"

python merge_peft_adapter.py \
    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/apfgoodandrandda/checkpoint-9000" \
    --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
    --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/apfrandcarto/"

python merge_peft_adapter.py \
    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/wgptgoodandrandda/checkpoint-10500" \
    --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
    --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/wgptrandcarto"

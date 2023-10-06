# Get version of the SFT Model
python merge_peft_adapter.py \
    --adapter_model_name="mnoukhov/llama-7b-se-peft" \
    --base_model_name="huggyllama/llama-7b" \
    --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft"

# Get version of the Original Reward Model
python merge_peft_adapter.py \
    --adapter_model_name="mnoukhov/llama-7b-se-rm-peft" \
    --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
    --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rmodel"

# Get version of what they get after PPO 
python merge_peft_adapter.py \
    --adapter_model_name="mnoukhov/llama-7b-se-rl-peft" \
    --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
    --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/ppomodel"
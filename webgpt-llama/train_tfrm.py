# Code for setting up (and training) a token-factored reward model

from transformers import (
    HfArgumentParser,
    TrainerCallback,
)

from rlhfutils.rmcode import (
    ScriptArguments, 
    get_trainargs, 
    load_tokenfactored_rmodel, 
    RewardDataCollatorWithPadding, 
    compute_metrics_tfr,
    TokenFactoredRewardTrainer
)
from rlhfutils.data import load_rm_wgpt

# parse args and load data
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

training_args = get_trainargs(script_args)

# NOTE this part changes to token-factored version
tokenizer, model = load_tokenfactored_rmodel(script_args)

train_dataset, eval_dataset = load_rm_wgpt(script_args, tokenizer)

# NOTE this part changes to TFRewardTrainer
# Train the model, woohoo.
trainer = TokenFactoredRewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics_tfr,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)

if script_args.eval_first_step:
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

trainer.train(script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
model.save_pretrained(script_args.output_dir + "_peft_last_checkpoint")
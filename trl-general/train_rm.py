#  General file for training RMs, compartmentalize as much as possible so that we can run all RMs for all settings through the same thing

# Code for setting up (and training) a token-factored reward model

from transformers import (
    HfArgumentParser,
    TrainerCallback,
)

from rlhfutils.rmcode import (
    ScriptArguments, 
    get_trainargs, 
    load_rmodel_standard, 
    RewardDataCollatorWithPadding, 
    compute_metrics,
    RewardTrainer
)
from rlhfutils.data import (
    load_wgpt,
    load_rlcd,
    load_stack,
    load_apfarm,
    tokenize_dset
)

# parse args and load data
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

training_args = get_trainargs(script_args)

tokenizer, model = load_rmodel_standard(script_args)

# Load in dataset according to params, we get something in format of 
if "wgpt" in script_args.dataset:
    train_dataset, eval_dataset = load_wgpt()
elif "rlcd" in script_args.dataset:
    train_dataset, eval_dataset = load_rlcd()
elif "stack" in script_args.dataset:
    train_dataset, eval_dataset = load_stack()
elif "apfarm" in script_args.dataset:
    train_dataset, eval_dataset = load_apfarm(script_args.dataset)

# TODO apply different kinds of data augmentation


# TODO will need to do template application stuff for this I think
# tokenize the dataset
train_dataset, eval_dataset = tokenize_dset(train_dataset, eval_dataset, script_args, tokenizer)
# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
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
    
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import pandas as pd


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )
    output_dir: Optional[str] = field(
        default="checkpoints/wgptsaved"
    )


# NOTE process anthro hh data for mixture with se data
def preproc_wgpt(example):
    ex = {}
    ex['question'] = example['question']['full_text']
    if example['score_0']>example['score_1']:
        ex['response_j'] = example['answer_0']
        ex['response_k'] = example['answer_1']
    else:
        ex['response_k'] = example['answer_0']
        ex['response_j'] = example['answer_1']
    return ex

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the human stack-exchange-paired dataset for tuning the reward model.
# HACK using wrong kind of data

#hh_train = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
#hh_train = hh_train.map(preproc_hh)
train_dataset = load_dataset("openai/webgpt_comparisons", split="train")
train_dataset = train_dataset.map(preproc_wgpt)

print("initial size ", len(train_dataset))
train_dataset = train_dataset.shuffle(seed=100)

# take the last portion as a test set
eval_dataset = train_dataset.select(range(18000, len(train_dataset)))
# use the rest as necessary
train_dataset = train_dataset.select(range(18000))

print("new size ", len(train_dataset))

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
output_name = script_args.output_dir

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)
# Load the value-head model and tokenizer.
tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

def len_balance(indataset):
    # Convert the Hugging Face dataset to a pandas DataFrame
    dataset_df = pd.DataFrame(indataset)

    # Create a new column to identify which row has longer input_ids_j
    dataset_df['j_is_longer'] = dataset_df.apply(lambda row: len(row['input_ids_j']) > len(row['input_ids_k']), axis=1)

    # Count how many examples are in each class
    count_j_is_longer = len(dataset_df[dataset_df['j_is_longer'] == True])
    count_k_is_longer = len(dataset_df) - count_j_is_longer

    # Determine which class has fewer examples
    min_count = min(count_j_is_longer, count_k_is_longer)

    # Downsample the class with more examples to have as many as the class with fewer examples
    if count_j_is_longer > min_count:
        df_j_is_longer = dataset_df[dataset_df['j_is_longer'] == True].sample(n=min_count)
        df_k_is_longer = dataset_df[dataset_df['j_is_longer'] == False]
    else:
        df_k_is_longer = dataset_df[dataset_df['j_is_longer'] == False].sample(n=min_count)
        df_j_is_longer = dataset_df[dataset_df['j_is_longer'] == True]

    # Concatenate the two downsampled classes to create a balanced DataFrame
    balanced_df = pd.concat([df_j_is_longer, df_k_is_longer])

    # Optionally, shuffle the rows
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    return  Dataset.from_pandas(balanced_df)


# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)
print("before balancing ", len(train_dataset))
# HACK doing length balancing here
train_dataset = len_balance(train_dataset)

print("after", len(train_dataset))

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


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
model.save_pretrained(output_name + "_peft_last_checkpoint")
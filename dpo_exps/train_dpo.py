# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from accelerate import Accelerator

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM

from trl import DPOTrainer
from rlhfutils.data import load_rlcd, load_wgpt, load_stack, inp_origformat, adjust_apf, load_manual

os.environ["WANDB_TAGS"] = "[\"dporlhf\"]"

#HACK currently modified DPO logging code

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "the name of the dataset we're doing"},
    )
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=300, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=600, metadata={"help": "the maximum sequence length"})
    epochs: Optional[int] = field(default=10, metadata={"help": "max number of epochs"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type: sigmoid, ipo, kto"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    evaldata: Optional[str] = field(
        default=None,
        metadata={"help": "the name of the dataset we're doing"},
    )

def tulu_pf(question, answer):
    return "<user>\n"+question+"\n<assistant>\n"+answer
    
def load_dpo_data(
    script_args,
    dataset: str = None,
    eval_dataset: str=None,
    num_proc=12,
) -> Dataset:   
    # Load in data from the right datasets
    if dataset == 'wgpt':
        train_data, eval_data = load_wgpt()
        pfunct = adjust_apf
    elif dataset == 'stack': 
        train_data, eval_data = load_stack()
        pfunct = inp_origformat
    elif dataset == 'rlcd':
        train_data, eval_data = load_rlcd()
        pfunct = adjust_apf
    else: 
        train_data, eval_data = load_manual(dataset, "", testdir=eval_dataset)
        pfunct = adjust_apf
    
    if "tulu" in script_args.model_name_or_path:
        pfunct = tulu_pf
        
    # TODO add in a sanity check

    original_columns = train_data.column_names
    eval_columns = eval_data.column_names
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            # PFUNCT is variable based on the base model (question: answer: for stack, ##instruction when using apf base model)
            "prompt": [pfunct(question, "") for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    final_train = train_data.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )    
    final_eval = eval_data.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=eval_columns,
    )
    # print("len before filter ", len(final_train))
    
    # final_train = final_train.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )
    # print("len after filter", len(final_train))
    # final_eval = final_eval.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )
    return final_train, final_eval

# TODO clean up this code so it isn't segmented by datasets
def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
        
    # TODO add in a sanity check

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map={"": Accelerator().local_process_index},
        load_in_8bit=True, 
        torch_dtype=torch.bfloat16
    )
    
    # model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model_ref = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     device_map={"": Accelerator().local_process_index},
    #     load_in_8bit=True
    # )
    # NOTE changed tokenizer path hardcoding
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # NOTE we're now doing this model
    train_dataset, eval_dataset = load_dpo_data(script_args, script_args.dataset, eval_dataset=script_args.evaldata)
    
    print("length of dataset is ", len(train_dataset))
    
    # 2. Load the Stack-exchange paired dataset
    # oldstack = get_stack_exchange_paired(data_dir="data/rl", sanity_check=script_args.sanity_check)
    # oldstack = train_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )
    
    # print("MAKE SURE TO LOOK AT THIS DATA AND MAKE SURE IT MAKES SENSE")
    # if script_args.dataset=='stack':
    #     print('previous data loading logic')
    #     print(oldstack[0]['prompt'])
    #     print(oldstack[0]['chosen'])
    
    print("TRAIN DATA")
    print(train_dataset[0]['prompt'])
    print(train_dataset[0]['chosen'])
    print(train_dataset[0]['rejected'])
    print("EVAL DATA")
    print(eval_dataset[0]['prompt'])
    print(eval_dataset[0]['chosen'])
    print(eval_dataset[0]['rejected'])
    
    # # 3. Load evaluation dataset
    # eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation", sanity_check=True)
    
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="trl_dpo",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        loss_type=script_args.loss_type
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
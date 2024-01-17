from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple 

import evaluate
import numpy as np
import torch
from torch import nn
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    LlamaPreTrainedModel,
    LlamaModel
)
import pickle
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import PaddingStrategy
import random 
from peft import PeftModel
import json

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
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "name of dataset, used to figure out preprocessing / data loading stuff"
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=1e-5)
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
    tokenbased: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use the token prior RM setup instead"
        },
    )
    num_train_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    eval_steps: Optional[int] = field(
        default=100,
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
    carto_file: Optional[str] = field(
        default=None,
        metadata={"help": "File storing indices of subset after dataset cartography"},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run eval after the first step"},
    )
    use_magnitude: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run eval after the first step"},
    )
    output_dir: Optional[str] = field(
        default="checkpoints/wgptsaved"
    )
    mix_ratio: Optional[float] = field(
        default=0, #HACK used to be 0.2, make sure to switch back at some point
        metadata={"help": "How much HH data to mix in"},
    )
    rand_ratio: Optional[float] = field(
        default=0, #HACK used to be 0.2, make sure to switch back at some point
        metadata={"help": "How much random augmentation to do"},
    )
    balance_len: Optional[int] = field(default=0)
    evaldata: Optional[str] = field(
        default=None,
        metadata={"help": "the name of the dataset we're using for eval"},
    )
    vardata: Optional[str] = field(
        default=None,
        metadata={"help": "if we want to do variance-maxing updates"},
    )
    # TO add preference over preference loss, make sure to have prefoverpref in output_dir name
    # pref_o_pref: Optional[int] = field(default=0) # use preference over preference loss
    
def get_trainargs(script_args):
    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    output_name = script_args.output_dir

    rname = output_name.split("/")
    try:
        rname = rname[-1] if output_name[-1]!='/' else rname[-2]
    except:
        rmname = "nothing"
    return TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.eval_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        include_inputs_for_metrics=False,
        run_name=rname, # wandb based on name set for output
        
    )
    
def load_rmodel_standard(script_args):
    # Load the value-head model and tokenizer.
    tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS if script_args.tokenbased else TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    modtype =  AutoModelForTokenClassification if script_args.tokenbased else AutoModelForSequenceClassification
    if "70b" in script_args.model_name: 
        # maybe see what happens with this thing in 8bit? 
        model = modtype.from_pretrained(
            script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, load_in_4bit=True#, device_map='auto'
        )
        model.config.pretraining_tp = 1 
    else:
        model = modtype.from_pretrained(
            script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16,# device_map="auto"
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    make_folder_if_not_exists("carto_outs/")
    model.config.use_cache = not script_args.gradient_checkpointing
    return tokenizer, model

# Same as above but this time it's for a TFR Reward Model
def load_tokenfactored_rmodel(script_args):
    # Load the value-head model and tokenizer.
    tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        # NOTE this changes from SEQ_CLS
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # NOTE token classification instead of sequence classification
    tmp = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16 # HACK just for denali
    )
    model = LlamaForTokenClassification(seqmodel=tmp, config=tmp.config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not script_args.gradient_checkpointing
    return tokenizer, model
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
            "ids": [f["ids"] for f in features], # propagate datapoint ID through code for data carto
            "return_loss": True,
        }
        # add logic for using mag in training
        if "mag" in features[0].keys():
            batch['mag'] = torch.tensor([f["mag"] for f in features])
        return batch
    
import os

def make_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

class RewardTrainer(Trainer):

    def save_carto(self, inps, rewards_j, rewards_k):
        fname = self.args.output_dir.strip().split("/")
        if len(fname[-1])==0:
            fname = fname[-2]
        else:
            fname = fname[-1]
        
        savef = open("carto_outs/"+fname+".jsonl", "a")  # append mode
        #print(rewards_j.shape)
        for i in range(len(inps['ids'])):
            td = {"uid":inps['ids'][i], "rew_j":float(rewards_j[i]), "rew_k":float(rewards_k[i])}
            savef.write(json.dumps(td)+"\n")
        savef.close()

    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs["input_ids_j"].requires_grad)
        
        if self.script_args.tokenbased:

            # Assuming `model` is a token classification model
            outputs_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])
            outputs_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])

            probs_j = torch.sigmoid(outputs_j.logits)
            probs_k = torch.sigmoid(outputs_k.logits)
            
            rewards_j = probs_j.sum(dim=1)
            rewards_k = probs_k.sum(dim=1)
        else:
            rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
            rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        
        mg = torch.zeros_like(rewards_j)
        # need to use mag key to actually use magnitude, TODO can just make this an arg
        if "mag" in inputs.keys() and "mag" in self.args.output_dir:
            # note that the shape should be the same? 
            assert mg.shape==inputs['mag'].unsqueeze(-1).shape
            mg = inputs["mag"].unsqueeze(-1)
            
            # print('indeed we are using magnitude loss')
        rdiff = rewards_j - rewards_k
        # NOTE adding in magnitude based loss
        loss = -nn.functional.logsigmoid(rdiff - mg).mean()
        # do preference over preference loss too
        if "prefoverpref" in self.args.output_dir:
            assert mg.shape[1]==1
            assert rdiff.shape[1]==1
            for i in range(0, len(rewards_j)-1):
                for j in range(i+1, len(rewards_j)):
                    if i==j:
                        continue
                    # more loss for pref over pref
                    if mg[i][0]>[j][0]:
                        loss = loss + -nn.functional.logsigmoid(rdiff[i][0] - rdiff[j][0] - (mg[i][0]-mg[j][0])).mean()
                    elif mg[j][0]>mg[i][0]:
                        loss = loss + -nn.functional.logsigmoid(rdiff[j][0] - rdiff[i][0] - (mg[j][0]-mg[i][0])).mean()
        
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        self.save_carto(inputs, rewards_j, rewards_k)
        return loss

def get_starts(seq):
    stmps = (seq == 22550).nonzero(as_tuple=True)
    if len(stmps[0])==len(seq):
        return stmps[1]+2
    starts = torch.zeros(seq.shape[0], 1).int()
    ind = 0
    for s in range(len(stmps[0])):
        if ind==len(starts):
            break
        if starts[ind]==0 and stmps[0][s]==ind:
            starts[ind] = stmps[1][s]+2
            ind = ind+1
    return starts.squeeze()

def amask(seq):
    mask = torch.zeros_like(seq)
    jstarts = get_starts(seq)
    for i in range(len(jstarts)):
        mask[i, jstarts[i]:] = 1
    return mask
    
class TokenFactoredRewardTrainer(Trainer):
    rounds = 1
    sublen = 25
                
    # given batch for j and k, return sampled subseqs to use for loss
    def sample_tokmasks(self, sequence_j, sequence_k, sublen):
        # find starting inds of sequence
        jstarts = get_starts(sequence_j).to(sequence_j.device)
        kstarts = get_starts(sequence_k).to(sequence_j.device)
        jmasks = torch.zeros_like(sequence_j).to(sequence_j.device)
        kmasks = torch.zeros_like(sequence_k).to(sequence_j.device)
        # find how long each seq is
        minlens = torch.min(sequence_j.shape[1]-jstarts, sequence_k.shape[1]-kstarts)
        for i in range(len(jstarts)):
            iks = kstarts[i].clone()
            if minlens[i]<sublen:
                kmasks[i, kstarts[i]:kstarts[i]+sublen] = 1
                jmasks[i, jstarts[i]:jstarts[i]+sublen] = 1
                continue
            kstarts[i] = random.randint(int(kstarts[i]), int(kstarts[i]+minlens[i]-sublen))
            kmasks[i, kstarts[i]:kstarts[i]+sublen] = 1
            jstarts[i] = random.randint(int(jstarts[i]+kstarts[i]-iks), int(jstarts[i]+minlens[i]-sublen))
            jmasks[i, jstarts[i]:jstarts[i]+sublen] = 1
        # sample starts 
        return kmasks, jmasks
    
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # TODO we need to check the shape 
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0].squeeze()
        # print(rewards_j.shape)
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0].squeeze()
        loss_list = []
        # Sample mask 5 times, 
        for _ in range(self.rounds):
            with torch.no_grad():
                km, jm = self.sample_tokmasks(inputs["input_ids_j"], inputs["input_ids_k"], self.sublen)
            # NOTE ChatGPT was right, I just didn't realize it until it was too late...
            # TODO sanity check these masks at some point
            kreward = torch.sum(rewards_k*km, dim=-1)/torch.sum(km, dim=-1)
            jreward = torch.sum(rewards_j*jm, dim=-1)/torch.sum(jm, dim=-1)
            loss_list.append(-nn.functional.logsigmoid(jreward-kreward).mean())
            
        loss = torch.mean(torch.stack(loss_list))
            
        if return_outputs:
            jmask = amask(rewards_j)
            kmask = amask(rewards_k)
            return loss, {
                "rewards_j": torch.sum(rewards_j*jmask, -1)/torch.sum(jmask, dim=-1), 
                "rewards_k": torch.sum(rewards_k*kmask, -1)/torch.sum(kmask, dim=-1)
            }
        return loss
    
    # def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys: bool):
    #     rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0].squeeze()
    #     # print(rewards_j.shape)
    #     rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0].squeeze()
    #     return None, {
    #         "rewards_j": torch.mean(rewards_j*amask(rewards_j), -1), 
    #         "rewards_k": torch.mean(rewards_k*amask(rewards_k), -1)
    #     }, None
    
# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    with open('outputs/evalpickles/tmpmetric.pickle', 'wb') as file:
        pickle.dump(eval_pred, file)
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def compute_metrics_tfr(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def load_rm(basepath, checkpath, iseval=True):
    tmp = AutoModelForSequenceClassification.from_pretrained(
        basepath, num_labels=1, torch_dtype=torch.bfloat16 # HACK just for denali
    )
    model = LlamaForTokenClassification(seqmodel=tmp, config=tmp.config)
    model = PeftModel.from_pretrained(model, checkpath)
    if iseval:
        model.eval()
    return model.merge_and_unload()
    

class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config, seqmodel=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        if seqmodel:
            self.transformer = seqmodel.model
        # 0.1 dropout for classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # TODO maybe this is, isn't necessary?
        self.to(dtype=torch.bfloat16)
        # Model parallel
        # self.model_parallel = False
        # self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
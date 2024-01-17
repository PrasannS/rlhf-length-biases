import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch

def do_train(args):
    # 1. Load the dataset
    dataset = Dataset.load_from_disk(args.dataset)

    # 2. Process the dataset
    def preprocess_function(examples):
        tokinps =  tokenizer(examples['text'], truncation=True, padding='max_length',max_length=512, return_tensors='pt')
        tokinps['labels'] = examples['score']
        return tokinps

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # making sure that we're within length
    tokenized_dataset = dataset.shuffle(seed=0).map(preprocess_function, batched=True, num_proc=10)

    # 3. Define the model
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.eos_token_id

    # 4. Training with specified arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        eval_steps=500, 
        logging_steps=10
    )
    # start with this for now, TODO fix this up later
    ratio = 0.99
    train_dset = tokenized_dataset.select(range(int(len(tokenized_dataset)*ratio)))
    eval_dset = tokenized_dataset.select(range(int(len(tokenized_dataset)*ratio), len(tokenized_dataset)))
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
    )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for saving output and checkpoints')
    parser.add_argument('--base_model', type=str, default='facebook/opt-125m', help='Base model name')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--dataset', type=str, default=None, help='Where to load in the train dataset from')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_steps', type=int, default=5000, help='Save checkpoint every X steps')
    args = parser.parse_args()

    trainer = do_train(args)
    trainer.train()

    # 5. Evaluation
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()

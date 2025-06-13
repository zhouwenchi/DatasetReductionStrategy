# coding=utf-8
import argparse
import logging
import math
import os
import random
import torch
import evaluate
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from accelerate import Accelerator

logger = logging.getLogger(__name__)

task_to_keys = {
    #"ocnli": ("premise", "hypothesis")
    None: ("sentence1", "sentence2")

}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on OCNLI task")
    #parser.add_argument("--task_name", type=str, default=None, help="The name of the task to train on.")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task to train on.", choices=list(task_to_keys.keys()))
    
    
    parser.add_argument("--train_file", type=str, default=None, help="A csv or json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="A csv or json file containing the validation data.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="Pad all samples to `max_length` if passed.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier.",
                        required=True)
    parser.add_argument("--tokenizer_name", type=str, default="hfl/chinese-bert-wwm", help="Path to the tokenizer.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size for training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size for evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Steps to accumulate before backprop.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="Scheduler type to use.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Warmup steps in lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()
    accelerator = Accelerator()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)



    #raw_datasets = load_dataset("json", data_files={"train": args.train_file, "validation": args.validation_file})
    #raw_datasets = load_dataset("csv", data_files={"train": args.train_file, "validation": args.validation_file})
    

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)


    #label_list = raw_datasets["train"].unique("label")
    #label_list.sort()
    #num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)



    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        sentence1_key, sentence2_key = "sentence1", None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    
    
    padding = "max_length" if args.pad_to_max_length else False
      
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    """
    def preprocess_function(examples):
        texts = (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        result["labels"] = examples["label"]
        return result

    """
    def preprocess_function(examples):
    # Tokenize the texts
        #texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        texts = (examples.get(sentence1_key),examples.get(sentence2_key, None)) if sentence2_key is not None else (examples[sentence1_key],)
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
            # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
            # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result


    #processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, desc="Running tokenizer on dataset")
    processed_datasets = raw_datasets.map(preprocess_function,
                                          batched=True,
                                          remove_columns=raw_datasets["train"].column_names,
                                          desc="Running tokenizer on dataset")
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

# Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              eval_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                                 num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps)
    #metric = load_metric("accuracy")
    metric = evaluate.load("accuracy") 

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=accelerator.gather(predictions),
                             references=accelerator.gather(batch["labels"]))

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()

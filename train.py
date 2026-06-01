import copy
import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
    full_finetune: bool = False
    bits: int = 16
    use_lora: bool = False
    init_weights: bool = True
    adapter_name_or_path: Optional[str] = None

    lora_rank: int = 128
    lora_alpha: float = 128.0
    lora_dropout: float = 0.0

    data_path: str = None
    sub_task: Optional[List[str]] = None
    dataset_split: str = "train"
    dataset_field: List[str] = None
    shuffle_dataset: bool = False
    model_max_length: int = 512
    merge: bool = False


def _tokenize_fn(strings, tokenizer):
    tokenized = [tokenizer(text, max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    return dict(
        input_ids=[np.array(t.input_ids) for t in tokenized],
        input_ids_lens=[len(t.input_ids) for t in tokenized],
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(s, tokenizer) for s in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=i)) for i in examples[query]]
    targets = [f"{o}\n{tokenizer.eos_token}" for o in examples[response]]
    return preprocess(sources, targets, tokenizer)


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in labels], batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def build_model(args):
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    quantization_config = None
    if int(args.bits) == 4:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("4-bit training requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype if quantization_config is None else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.full_finetune:
        for p in model.parameters():
            p.requires_grad = True
        return model

    if int(args.bits) == 4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.adapter_name_or_path:
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
            init_lora_weights=bool(args.init_weights),
        )
        model = get_peft_model(model, config)

    model.print_trainable_parameters()
    return model


def build_dataset(args, tokenizer):
    if args.sub_task:
        datasets = []
        for task in args.sub_task:
            task_name, split_limit = (task.split(":", 1) if ":" in task else (task, None))
            split = f"{args.dataset_split}[:{split_limit}]" if split_limit else args.dataset_split
            datasets.append(load_dataset(args.data_path, data_dir=task_name, split=split))
        train_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    else:
        train_dataset = load_dataset(args.data_path, split=args.dataset_split)

    if args.shuffle_dataset:
        train_dataset = train_dataset.shuffle(seed=args.seed)

    return train_dataset.map(
        train_tokenize_function,
        batched=True,
        num_proc=32,
        remove_columns=train_dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": args.dataset_field[0],
            "response": args.dataset_field[1],
        },
    )


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    set_global_seed(int(args.seed))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args)
    train_dataset = build_dataset(args, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    trainer.train()

    if trainer.is_world_process_zero():
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        if args.merge and not args.full_finetune and hasattr(model_to_save, "merge_and_unload"):
            model_to_save = model_to_save.merge_and_unload()
            model_to_save.save_pretrained(args.output_dir)
        else:
            trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()

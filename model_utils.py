from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
import config

@dataclass
class QAModelConfig:
    model_name: str = config.MODEL_NAME
    learning_rate: float = config.TRAIN_CONFIG["learning_rate"]
    num_train_epochs: float = config.TRAIN_CONFIG["num_train_epochs"]
    per_device_train_batch_size: int = config.TRAIN_CONFIG["train_batch_size"]
    per_device_eval_batch_size: int = config.TRAIN_CONFIG["eval_batch_size"]
    weight_decay: float = config.TRAIN_CONFIG["weight_decay"]
    warmup_ratio: float = config.TRAIN_CONFIG["warmup_ratio"]
    logging_steps: int = 100
    save_total_limit: int = 1

def load_tokenizer(model_name: str = config.MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def load_qa_model(model_name: str = config.MODEL_NAME):
    return AutoModelForQuestionAnswering.from_pretrained(model_name)

def compute_squad_metrics(eval_pred):
    # Placeholder: We compute accurate EM/F1 offline using the main.py 
    # 'eval-baseline' command to avoid overhead during the training loop.
    return {}

def train_qa_model(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    output_dir: str,
    cfg: QAModelConfig,
):
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_qa_model(cfg.model_name)

    def data_collator(features):
        first = features[0]
        batch = {}
        for k in first.keys():
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
        return batch

    has_eval = eval_dataset is not None
    eval_strategy = "epoch" if has_eval else "no"
    save_strategy = "epoch" if has_eval else "no"

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=has_eval,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_squad_metrics, # Disabled for speed; see eval script
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer

def load_finetuned_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    return tokenizer, model

def _question_only_inputs(batch, tokenizer):
    return tokenizer(
        batch["question"],
        ["" for _ in batch["question"]],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

def _context_only_inputs(batch, tokenizer):
    return tokenizer(
        ["" for _ in batch["context"]],
        batch["context"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

@torch.no_grad()
def run_question_only_baseline(
    model_dir: str, eval_dataset: Dataset, num_examples: int = 200
):
    tokenizer, model = load_finetuned_model(model_dir)
    model.eval()

    subset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))
    scores = []

    for ex in subset:
        enc = _question_only_inputs({"question": [ex["question"]]}, tokenizer)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        outputs = model(**enc)
        start_logits = outputs.start_logits[0].softmax(dim=-1)
        end_logits = outputs.end_logits[0].softmax(dim=-1)
        scores.append(float(start_logits.max() * end_logits.max()))

    return float(np.mean(scores))

@torch.no_grad()
def run_context_only_baseline(
    model_dir: str, eval_dataset: Dataset, num_examples: int = 200
):
    tokenizer, model = load_finetuned_model(model_dir)
    model.eval()

    subset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))
    scores = []

    for ex in subset:
        enc = _context_only_inputs({"context": [ex["context"]]}, tokenizer)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        outputs = model(**enc)
        start_logits = outputs.start_logits[0].softmax(dim=-1)
        end_logits = outputs.end_logits[0].softmax(dim=-1)
        scores.append(float(start_logits.max() * end_logits.max()))

    return float(np.mean(scores))
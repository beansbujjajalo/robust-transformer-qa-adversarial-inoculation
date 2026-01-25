import json
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from data_utils import prepare_train_features
from model_utils import QAModelConfig, load_tokenizer, train_qa_model
import config

logger = logging.getLogger(__name__)

def build_adversarial_train_subset(dataset: DatasetDict, num_examples: int = 5000, seed: int = config.SEED,) -> List[Dict]:
    """
    Constructs an adversarial training set by appending distractor sentences.
    Ref: Jia & Liang (2017) 'Adversarial SQuAD'.
    """
    random.seed(seed)
    train = dataset["train"]
    indices = list(range(len(train)))
    random.shuffle(indices)
    indices = indices[: min(num_examples, len(indices))]

    adv_examples = []
    logger.info(f"Generating {len(indices)} adversarial examples...")
    
    for idx in indices:
        ex = train[idx]
        base_context = ex["context"]
        # Use random distractor from config pool (Report Section 3.2)
        distractor = random.choice(config.ADVERSARIAL_DISTRACTORS)
        context_adv = base_context + " " + distractor
        
        adv_examples.append(
            {
                "id": f"adv-{ex['id']}",
                "title": ex.get("title", ""),
                "context": context_adv,
                "question": ex["question"],
                "answers": ex["answers"],
            }
        )
    return adv_examples

def save_adversarial_examples(examples: List[Dict], path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

def load_adversarial_examples(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def build_robust_train_dataset(base_dataset: DatasetDict, adv_dataset: Dataset, tokenizer_name_or_dir: str,) -> Tuple[Dataset, Dataset]:
    tokenizer = load_tokenizer(tokenizer_name_or_dir)

    tokenized_train_orig = base_dataset["train"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=base_dataset["train"].column_names,
    )

    tokenized_train_adv = adv_dataset.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=adv_dataset.column_names,
    )

    combined_train = concatenate_datasets(
        [tokenized_train_orig, tokenized_train_adv]
    )

    tokenized_val = base_dataset["validation"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=base_dataset["validation"].column_names,
    )

    return combined_train, tokenized_val

@dataclass
class RobustTrainingConfig(QAModelConfig):
    adv_weight: float = 1.0

def run_adversarial_finetuning(baseline_dir: str, output_dir: str, adv_examples_path: str, num_train_epochs: float = 1.0,):
    dataset = load_dataset("squad")
    adv_dataset = load_adversarial_examples(adv_examples_path)

    train_dataset, val_dataset = build_robust_train_dataset(
        dataset, adv_dataset, tokenizer_name_or_dir=baseline_dir
    )

    cfg = RobustTrainingConfig(
        model_name=baseline_dir,
        num_train_epochs=num_train_epochs,
    )

    trainer = train_qa_model(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=output_dir,
        cfg=cfg,
    )
    return trainer

def build_adversarial_split(split: Dataset, num_examples: int = 2000, seed: int = 123,) -> Dataset:
    random.seed(seed)
    indices = list(range(len(split)))
    random.shuffle(indices)
    indices = indices[: min(num_examples, len(indices))]

    adv_examples = []
    for idx in indices:
        ex = split[idx]
        base_context = ex["context"]
        distractor = random.choice(config.ADVERSARIAL_DISTRACTORS)
        context_adv = base_context + " " + distractor
        
        adv_examples.append(
            {
                "id": f"dev-adv-{ex['id']}",
                "title": ex.get("title", ""),
                "context": context_adv,
                "question": ex["question"],
                "answers": ex["answers"],
            }
        )

    return Dataset.from_list(adv_examples)
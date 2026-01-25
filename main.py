import argparse
import os
import logging
import random
import torch
import numpy as np
from datasets import load_dataset, Dataset
import config

from data_utils import (
    load_squad,
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
)
from model_utils import (
    QAModelConfig,
    load_tokenizer,
    load_finetuned_model,
    train_qa_model,
    run_question_only_baseline,
    run_context_only_baseline,
)
from robustness_utils import (
    build_adversarial_train_subset,
    save_adversarial_examples,
    run_adversarial_finetuning,
    build_adversarial_split,
)
from checklist_utils import run_checklist, format_checklist_results

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Enforce reproducibility for report results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_em_f1(examples: Dataset, predictions: dict):
    """
    Computes Exact Match and F1 scores.
    IMPL NOTE: Uses the official SQuAD v1.1 normalization logic (Rajpurkar et al., 2016)
    to ensure comparability with standard benchmarks.
    """
    import string
    import re

    def normalize_text(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

    def f1_score(prediction, ground_truth):
        pred_tokens = normalize_text(prediction).split()
        gt_tokens = normalize_text(ground_truth).split()
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return float(pred_tokens == gt_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def exact_match_score(prediction, ground_truth):
        return float(normalize_text(prediction) == normalize_text(ground_truth))

    em_scores = []
    f1_scores = []

    for ex in examples:
        qid = ex["id"]
        if qid not in predictions:
            continue
        pred = predictions[qid]
        gold_answers = ex["answers"]["text"]
        if len(gold_answers) == 0:
            gold_answers = [""]
        em = max(exact_match_score(pred, ga) for ga in gold_answers)
        f1 = max(f1_score(pred, ga) for ga in gold_answers)
        em_scores.append(em)
        f1_scores.append(f1)

    return 100 * float(np.mean(em_scores)), 100 * float(np.mean(f1_scores))

def _eval_on_features(model_dir: str, examples: Dataset, features):
    import torch
    from torch.utils.data import DataLoader

    tokenizer, model = load_finetuned_model(model_dir)

    features_for_model = features.remove_columns(["offset_mapping", "example_id"])
    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in features_for_model.column_names:
        cols.append("token_type_ids")
    features_for_model.set_format(type="torch", columns=cols)

    model.eval()
    all_start_logits = []
    all_end_logits = []

    dl = DataLoader(features_for_model, batch_size=16)
    for batch in dl:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        raw_predictions=(all_start_logits, all_end_logits),
        tokenizer=tokenizer,
    )
    return compute_em_f1(examples, predictions)

def eval_model_on_squad(model_dir: str):
    dataset = load_dataset("squad")
    val_examples = dataset["validation"]

    tokenizer, _ = load_finetuned_model(model_dir)
    val_features = val_examples.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=val_examples.column_names,
    )

    em, f1 = _eval_on_features(model_dir, val_examples, val_features)
    logger.info(f"SQuAD dev – EM: {em:.2f}, F1: {f1:.2f}")
    return em, f1

def eval_model_on_adversarial_dev(model_dir: str, num_examples: int = 2000):
    dataset = load_dataset("squad")
    dev_split = dataset["validation"]
    adv_dev = build_adversarial_split(dev_split, num_examples=num_examples)

    tokenizer, _ = load_finetuned_model(model_dir)
    adv_features = adv_dev.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=adv_dev.column_names,
    )

    em, f1 = _eval_on_features(model_dir, adv_dev, adv_features)
    logger.info(f"Adversarial dev ({len(adv_dev)} ex) – EM: {em:.2f}, F1: {f1:.2f}")
    return em, f1

def cmd_train_baseline(args):
    dataset = load_squad()
    tokenizer = load_tokenizer()

    tokenized_train = dataset["train"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_val = dataset["validation"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    cfg = QAModelConfig(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
    )

    train_qa_model(
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        output_dir=args.output_dir,
        cfg=cfg,
    )

def cmd_eval_baseline(args):
    logger.info("Evaluating baseline model on SQuAD dev...")
    em, f1 = eval_model_on_squad(args.model_dir)
    print(f"Baseline – EM {em:.2f}, F1 {f1:.2f}")

    dataset = load_dataset("squad")
    q_conf = run_question_only_baseline(args.model_dir, dataset["validation"])
    c_conf = run_context_only_baseline(args.model_dir, dataset["validation"])
    print(f"Question-only avg max prob: {q_conf:.4f}")
    print(f"Context-only avg max prob:  {c_conf:.4f}")

def cmd_build_adversarial(args):
    dataset = load_squad()
    adv_examples = build_adversarial_train_subset(
        dataset, num_examples=args.num_examples
    )
    save_adversarial_examples(adv_examples, args.out_path)
    logger.info(f"Saved {len(adv_examples)} adversarial examples to {args.out_path}")

def cmd_train_robust(args):
    os.makedirs(args.output_dir, exist_ok=True)
    run_adversarial_finetuning(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        adv_examples_path=args.adv_path,
        num_train_epochs=args.num_train_epochs,
    )

def cmd_eval_robust(args):
    logger.info("Evaluating robust model on SQuAD dev...")
    em, f1 = eval_model_on_squad(args.model_dir)
    print(f"Robust – EM {em:.2f}, F1 {f1:.2f}")

def cmd_eval_adv_dev(args):
    logger.info("Evaluating model on adversarial dev split...")
    em, f1 = eval_model_on_adversarial_dev(
        args.model_dir, num_examples=args.num_examples
    )
    print(f"Adv dev – EM {em:.2f}, F1 {f1:.2f}")

def cmd_run_checklist(args):
    results = run_checklist(args.model_dir)
    print(format_checklist_results(results))

def main():
    set_seed(config.SEED)

    parser = argparse.ArgumentParser(
        description="SQuAD Robust QA project (dataset artifacts)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train-baseline")
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--num_train_epochs", type=float, default=2.0)
    p_train.add_argument("--batch_size", type=int, default=12)
    p_train.set_defaults(func=cmd_train_baseline)

    p_eval_base = subparsers.add_parser("eval-baseline")
    p_eval_base.add_argument("--model_dir", type=str, required=True)
    p_eval_base.set_defaults(func=cmd_eval_baseline)

    p_adv = subparsers.add_parser("build-adversarial")
    p_adv.add_argument("--out_path", type=str, required=True)
    p_adv.add_argument("--num_examples", type=int, default=5000)
    p_adv.set_defaults(func=cmd_build_adversarial)

    p_train_robust = subparsers.add_parser("train-robust")
    p_train_robust.add_argument("--baseline_dir", type=str, required=True)
    p_train_robust.add_argument("--output_dir", type=str, required=True)
    p_train_robust.add_argument("--adv_path", type=str, required=True)
    p_train_robust.add_argument("--num_train_epochs", type=float, default=1.0)
    p_train_robust.set_defaults(func=cmd_train_robust)

    p_eval_robust = subparsers.add_parser("eval-robust")
    p_eval_robust.add_argument("--model_dir", type=str, required=True)
    p_eval_robust.set_defaults(func=cmd_eval_robust)

    p_eval_adv = subparsers.add_parser("eval-adv-dev")
    p_eval_adv.add_argument("--model_dir", type=str, required=True)
    p_eval_adv.add_argument("--num_examples", type=int, default=2000)
    p_eval_adv.set_defaults(func=cmd_eval_adv_dev)

    p_cl = subparsers.add_parser("run-checklist")
    p_cl.add_argument("--model_dir", type=str, required=True)
    p_cl.set_defaults(func=cmd_run_checklist)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
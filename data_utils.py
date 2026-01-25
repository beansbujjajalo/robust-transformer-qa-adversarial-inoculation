import collections
from typing import Dict, Tuple

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
import config

def load_squad() -> DatasetDict:
    return load_dataset("squad")

def prepare_train_features(
    examples: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = config.MAX_SEQ_LENGTH,
    doc_stride: int = config.DOC_STRIDE,
) -> Dict:
    """
    Tokenizes examples with sliding window (stride) to handle long contexts.
    
    IMPL NOTE: Logic adapted from standard Hugging Face SQuAD preprocessing 
    (transformers/examples/pytorch/question-answering/run_qa.py) to ensure 
    correct handling of character-to-token offset mapping.
    """
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized["offset_mapping"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)

        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        answer_start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]

        context_start = None
        context_end = None
        for j, s_id in enumerate(sequence_ids):
            if s_id == 1 and context_start is None:
                context_start = j
            if s_id == 1:
                context_end = j

        if context_start is None or context_end is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        if not (
            offsets[context_start][0] <= answer_start_char
            and offsets[context_end][1] >= answer_start_char + len(answer_text)
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        token_start_index = context_start
        token_end_index = context_end

        while (
            token_start_index <= context_end
            and offsets[token_start_index][0] <= answer_start_char
            and offsets[token_start_index][1] <= answer_start_char
        ):
            token_start_index += 1
        token_start_index -= 1

        answer_end_char = answer_start_char + len(answer_text)
        while (
            token_end_index >= context_start
            and offsets[token_end_index][1] >= answer_end_char
            and offsets[token_end_index][0] >= answer_end_char
        ):
            token_end_index -= 1
        while (
            token_end_index <= context_end
            and offsets[token_end_index][1] < answer_end_char
        ):
            token_end_index += 1

        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")
    return tokenized

def prepare_validation_features(
    examples: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = config.MAX_SEQ_LENGTH,
    doc_stride: int = config.DOC_STRIDE,
) -> Dict:
    """
    Preprocessing for validation. Maintains overflow mappings to reconstruct
    predictions for the original example IDs.
    """
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx])

        offset = tokenized["offset_mapping"][i]
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offset)
        ]

    tokenized["example_id"] = example_ids
    return tokenized

def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: Tuple,
    tokenizer: PreTrainedTokenizerBase,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        features_per_example[example_id_to_index[feat["example_id"]]].append(i)

    predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        valid_answers = []
        context = example["context"]

        for feat_idx in feature_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]
            offsets = features[feat_idx]["offset_mapping"]

            start_indexes = sorted(
                range(len(start_logits)),
                key=lambda i: start_logits[i],
                reverse=True,
            )[:n_best_size]
            end_indexes = sorted(
                range(len(end_logits)),
                key=lambda i: end_logits[i],
                reverse=True,
            )[:n_best_size]

            for s in start_indexes:
                for e in end_indexes:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue

                    start_char = offsets[s][0]
                    end_char = offsets[e][1]
                    text = context[start_char:end_char]
                    score = start_logits[s] + end_logits[e]
                    valid_answers.append({"score": score, "text": text})

        if valid_answers:
            best = max(valid_answers, key=lambda x: x["score"])
            predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""

    return predictions
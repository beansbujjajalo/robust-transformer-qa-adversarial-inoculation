"""
Project Configuration and Hyperparameters.
Centralizes constants for reproducibility and citation.
"""

# Random seed for reproducibility (Sections 2.1, 3.2, 4.2)
# Ensures train/dev/adversarial splits are identical to reported results.
SEED = 42

# Training Hyperparameters (Report Section 3.1)
TRAIN_CONFIG = {
    "learning_rate": 3e-5,
    "num_train_epochs": 2,          # Baseline
    "robust_epochs": 1,             # Robust fine-tuning
    "train_batch_size": 12,
    "eval_batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}

# Model Architecture (Report Section 2.2)
MODEL_NAME = "google/electra-small-discriminator"

# SQuAD Preprocessing Constants
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128

# Adversarial Distractor Templates (Report Section 3.2 & 8.1)
# Methodology based on 'AddSent' (Jia and Liang, 2017).
# These are intentionally generic to probe for surface-level artifacts without 
# requiring external knowledge bases.
ADVERSARIAL_DISTRACTORS = [
    "However, an unrelated report mentioned a completely different person in another city.",
    "Some sources falsely claimed a different date, which is not correct.",
    "In another book, a similar event involved other people and took place elsewhere.",
    "There was also a old rumor that suggested an alternative explanation, but it turned out to be wrong.",
]
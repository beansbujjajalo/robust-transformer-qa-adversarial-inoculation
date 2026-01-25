````md
# Robust Transformer-Based QA via Adversarial Data Augmentation and Inoculation Fine-Tuning

Course project for **NATURAL LANGUAGE PROCESSING (Fall 2025)**.

This repo fine-tunes a small Transformer QA model on **SQuAD v1.1**, probes it for **dataset artifacts**, and then applies **adversarial / inoculation-style fine-tuning** to improve robustness against **distractor-augmented contexts**.

## What this project does

- Train a **baseline** ELECTRA-small model on SQuAD  
- Measure **question-only** and **context-only** baselines (artifact probes)  
- Build an **adversarial training set** by appending distractor sentences to contexts  
- Fine-tune a **robust** model on a mixture of original + adversarial data  
- Evaluate both models on:
  - Clean SQuAD dev
  - An **adversarial dev** split (distractor-augmented dev)
  - A tiny **CheckList-style** behavioral suite (negation, distractor, entity swap)

The pipeline is designed to be reusable for other extractive QA datasets.

---

## Results (from the project report)

| Setting | Model | EM | F1 |
|---|---:|---:|---:|
| Clean SQuAD dev | Baseline | 29.62 | 72.75 |
| Clean SQuAD dev | Robust | 30.06 | 73.73 |
| Adversarial dev (2,000 ex) | Baseline | 29.95 | 72.92 |
| Adversarial dev (2,000 ex) | Robust | 30.55 | 73.69 |

Artifact probes (avg max start×end prob over 200 dev examples):
- Question-only: 0.1664  
- Context-only: 0.0722  

> Note: numbers can shift slightly with hardware / library versions; trends should remain consistent.

---

## 1. Project structure

Key files:

- `main.py` – CLI entry point: `train-baseline`, `eval-baseline`, `build-adversarial`, `train-robust`, `eval-robust`, `eval-adv-dev`, `run-checklist`  
- `config.py` – seed, hyperparameters, model name, and distractor templates  
- `data_utils.py` – SQuAD loading + preprocessing + post-processing predictions  
- `model_utils.py` – model/tokenizer loading, Hugging Face `Trainer`, question-only/context-only probes  
- `robustness_utils.py` – adversarial train/dev construction + adversarial fine-tuning helpers  
- `checklist_utils.py` – tiny CheckList-style suite (6 examples, 3 categories)  
- `requirements.txt` – Python package requirements  

---

## 2. Requirements

Tested with:
- Python 3.10+  
- A **GPU** is strongly recommended (CPU training is very slow)
- Internet access (downloads model + SQuAD)

Install dependencies:

```bash
pip install -r requirements.txt
````

Main packages:

* `datasets`, `transformers`, `torch`, `accelerate`, `tqdm`, `numpy`, `scikit-learn`, `matplotlib`

---

## 3. Dataset (not included in this repo)

This repo **does not include SQuAD or other large datasets**.

SQuAD is loaded via Hugging Face:

* `datasets.load_dataset("squad")`

Adversarial data (distractor-augmented) is **generated locally** using `build-adversarial` (see below).

---

## 4. Reproducibility and configuration

Core settings live in `config.py`:

* `SEED = 42`
* `TRAIN_CONFIG` – LR, epochs, batch sizes, weight decay, warmup ratio
* `MODEL_NAME` – default: `google/electra-small-discriminator`
* `MAX_SEQ_LENGTH`, `DOC_STRIDE` – SQuAD windowing params
* `ADVERSARIAL_DISTRACTORS` – distractor sentences appended to contexts

---

## 5. How to run the full pipeline

All commands assume your working directory contains `main.py`.

### 5.1 Train the baseline model

```bash
python main.py train-baseline --output_dir runs/baseline --num_train_epochs 2 --batch_size 12
```

### 5.2 Evaluate baseline + artifact probes

```bash
python main.py eval-baseline --model_dir runs/baseline
```

Outputs:

1. SQuAD dev **EM/F1**
2. **question-only** probe
3. **context-only** probe

### 5.3 Build adversarial training data (5,000 examples)

```bash
python main.py build-adversarial --out_path data/adversarial_train.json --num_examples 5000
```

### 5.4 Inoculation-style fine-tuning (robust model)

```bash
python main.py train-robust \
  --baseline_dir runs/baseline \
  --output_dir runs/robust \
  --adv_path data/adversarial_train.json \
  --num_train_epochs 1
```

### 5.5 Evaluate robust model (clean dev)

```bash
python main.py eval-robust --model_dir runs/robust
```

### 5.6 Evaluate on adversarial dev (created on-the-fly)

```bash
python main.py eval-adv-dev --model_dir runs/baseline --num_examples 2000
python main.py eval-adv-dev --model_dir runs/robust   --num_examples 2000
```

### 5.7 Run the CheckList-style tests (optional)

```bash
python main.py run-checklist --model_dir runs/baseline
python main.py run-checklist --model_dir runs/robust
```

---

## 6. Typical workflow

```bash
pip install -r requirements.txt

python main.py train-baseline --output_dir runs/baseline --num_train_epochs 2 --batch_size 12
python main.py eval-baseline  --model_dir runs/baseline

python main.py build-adversarial --out_path data/adversarial_train.json --num_examples 5000
python main.py train-robust --baseline_dir runs/baseline --output_dir runs/robust --adv_path data/adversarial_train.json --num_train_epochs 1

python main.py eval-robust   --model_dir runs/robust
python main.py eval-adv-dev  --model_dir runs/baseline --num_examples 2000
python main.py eval-adv-dev  --model_dir runs/robust   --num_examples 2000

python main.py run-checklist --model_dir runs/robust
```

---

## 7. Customization ideas

* Swap the base model: change `MODEL_NAME` in `config.py` (e.g., `bert-base-uncased`)
* Use stronger distractors: update `ADVERSARIAL_DISTRACTORS`
* Change clean/adv proportions: downsample or add weighting
* Plug in another dataset: replace `load_squad()` + post-processing for the new dataset

---

## 8. Citations / Acknowledgements

* SQuAD dataset (Rajpurkar et al.)
* ELECTRA (Clark et al.)
* Hugging Face Transformers + Datasets

---

## 9. Troubleshooting

* CPU training is slow → use a GPU runtime (e.g., Colab)
* Slight EM/F1 drift across environments is normal

```

---

## Two quick files I recommend adding

### `.gitignore` (important)
Add at least:
- `data/`
- `runs/`, `outputs/`, `checkpoints/`
- model artifacts: `*.bin`, `*.pt`, `*.ckpt`
- caches: `__pycache__/`, `.DS_Store`, `.pytest_cache/`, `.ipynb_checkpoints/`

### `LICENSE`
MIT is common unless your program requires otherwise.

---

If you want, I can also give you:
- a clean `.gitignore` tailored to your exact folder names (`runs/`, `data/`, etc.)
- a 1–2 sentence **repo About** text and the exact **GitHub Topics** list to paste into settings (matching your LinkedIn wording).
```

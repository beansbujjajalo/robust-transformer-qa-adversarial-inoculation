# Robust Question Answering under Dataset Artifacts and Adversarial Distractors

This project fine-tunes a small transformer QA model on SQuAD, probes it for **dataset artifacts**, and then applies a simple **adversarial / inoculation-style fine-tuning** to make it a bit more robust.

Concretely, it lets you:

- Train a **baseline** ELECTRA-small model on SQuAD  
- Measure **question-only** and **context-only** baselines (artifact probes)  
- Build an **adversarial training set** by appending distractor sentences to contexts  
- Fine-tune a **robust** model on the mixture of original + adversarial data  
- Evaluate both models on:
  - Clean SQuAD dev
  - An **adversarial dev** split (distractor-augmented dev)
  - A tiny **CheckList-style** suite (negation, distractor, entity swap)

The code is designed so that you can reuse the pipeline later on other QA datasets.

---

## 1. Project structure

Key files:

- `main.py` – command-line entry point. Defines subcommands like `train-baseline`, `eval-baseline`, `build-adversarial`, `train-robust`, `eval-adv-dev`, and `run-checklist`.  
- `config.py` – central config: seed, training hyperparameters, model name, and adversarial distractor templates.  
- `data_utils.py` – SQuAD loading and preprocessing (train/validation feature preparation, post-processing predictions).  
- `model_utils.py` – model + tokenizer loading, training loop (HuggingFace `Trainer`), and question-only / context-only probes.  
- `robustness_utils.py` – adversarial train/dev construction and adversarial fine-tuning utilities.  
- `checklist_utils.py` – tiny, static CheckList-style suite (6 examples, 3 categories) and evaluation helper.  
- `requirements.txt` – Python package requirements (datasets, transformers, torch, etc.).  

---

## 2. Requirements

Tested with:

- Python 3.10+  
- A machine with **GPU** is strongly recommended (training on CPU is very slow)
- Internet access (to download SQuAD and the model from Hugging Face)

Install dependencies:

```bash
pip install -r requirements.txt
```

The main packages are:

- `datasets` (SQuAD loading)
- `transformers` (ELECTRA model, Trainer)
- `torch` (PyTorch)
- `accelerate`, `tqdm`, `numpy`, `scikit-learn`, `matplotlib`

---

## 3. Reproducibility and configuration

All core settings live in `config.py`:

- `SEED = 42` – used in `main.py` to seed Python, NumPy, and PyTorch so that the train/dev splits and adversarial subsets are reproducible.  
- `TRAIN_CONFIG` – learning rate, epochs, batch sizes, weight decay, warmup ratio.  
- `MODEL_NAME` – default model (`google/electra-small-discriminator`).  
- `MAX_SEQ_LENGTH`, `DOC_STRIDE` – SQuAD max sequence length and sliding window stride.  
- `ADVERSARIAL_DISTRACTORS` – small list of generic distractor sentences that get appended to contexts for adversarial training/dev.  

If you want to tweak hyperparameters or swap the base model, the idea is: change it in `config.py`, not in multiple places in the code.

---

## 4. How to run the full pipeline

All commands below assume your working directory contains `main.py` (e.g., `robust_qa_project/main.py`).

### 4.1 Train the baseline model

Fine-tune ELECTRA-small on SQuAD:

```bash
python main.py train-baseline   --output_dir runs/baseline   --num_train_epochs 2   --batch_size 12
```

This will:

- Download SQuAD via `datasets.load_dataset("squad")`  
- Tokenize train + validation with sliding windows (`prepare_train_features`)  
- Train for 2 epochs with the configuration in `QAModelConfig`  
- Save the model + tokenizer into `runs/baseline/`  

You’ll see training and evaluation losses in the logs.

---

### 4.2 Evaluate baseline + artifact baselines

Run evaluation on the standard SQuAD dev set:

```bash
python main.py eval-baseline --model_dir runs/baseline
```

This does three things:

1. Computes **EM / F1** on SQuAD dev using the official normalization (lowercasing, removing punctuation/articles).  
2. Runs a **question-only** probe: feed only the question and an empty context, average the max start×end probability.  
3. Runs a **context-only** probe: feed an empty question and the full context, again average the max start×end probability.  

This is how you get the “artifact” numbers that go into the report.

---

### 4.3 Build adversarial training data

Next, construct an adversarial training subset by appending distractor sentences to some training contexts:

```bash
python main.py build-adversarial   --out_path data/adversarial_train.json   --num_examples 5000
```

Under the hood, this:

- Samples `num_examples` training examples from SQuAD  
- Appends a random sentence from `config.ADVERSARIAL_DISTRACTORS` to each context  
- Writes them to `data/adversarial_train.json` as a list of SQuAD-style dicts  

---

### 4.4 Adversarial fine-tuning (robust model)

Starting from the baseline checkpoint, fine-tune on the mixture of original + adversarial data:

```bash
python main.py train-robust   --baseline_dir runs/baseline   --output_dir runs/robust   --adv_path data/adversarial_train.json   --num_train_epochs 1
```

This:

- Loads SQuAD again and the adversarial JSON file  
- Tokenizes original train split and adversarial examples with the same `prepare_train_features` logic  
- Concatenates them into a single training dataset  
- Runs one epoch of additional training starting from `baseline_dir`  

The robust checkpoint is saved in `runs/robust/`.

---

### 4.5 Evaluate the robust model (clean dev)

```bash
python main.py eval-robust --model_dir runs/robust
```

This is a thin wrapper around the same SQuAD dev evaluation used for the baseline, just printing `Robust – EM ..., F1 ...`.  

---

### 4.6 Evaluate on adversarial dev

Compare how both models behave on a **distractor-augmented dev split**:

```bash
# Baseline on adversarial dev
python main.py eval-adv-dev --model_dir runs/baseline --num_examples 2000

# Robust on adversarial dev
python main.py eval-adv-dev --model_dir runs/robust --num_examples 2000
```

`eval-adv-dev` creates an adversarial dev split on the fly (same distractor logic as the train subset) and then reuses the standard EM/F1 evaluation.

---

### 4.7 Run the CheckList-style tests

Finally, you can sanity-check the model on a tiny behavioral suite:

```bash
python main.py run-checklist --model_dir runs/baseline
python main.py run-checklist --model_dir runs/robust
```

The suite is defined statically in `checklist_utils.py` (two examples each for **negation**, **distractor**, and **entity swap**).  

`run_checklist` prints accuracy per category, counting an example as correct if the predicted answer string contains the gold answer.  

---

## 5. Typical workflow

If you just want to reproduce the main experiments:

1. **Install deps**  
   `pip install -r requirements.txt`
2. **Train baseline**  
   `python main.py train-baseline --output_dir runs/baseline --num_train_epochs 2 --batch_size 12`
3. **Evaluate baseline + artifact probes**  
   `python main.py eval-baseline --model_dir runs/baseline`
4. **Build adversarial train data**  
   `python main.py build-adversarial --out_path data/adversarial_train.json --num_examples 5000`
5. **Train robust model**  
   `python main.py train-robust --baseline_dir runs/baseline --output_dir runs/robust --adv_path data/adversarial_train.json --num_train_epochs 1`
6. **Evaluate robust (clean dev)**  
   `python main.py eval-robust --model_dir runs/robust`
7. **Evaluate both on adversarial dev**  
   `python main.py eval-adv-dev --model_dir runs/baseline --num_examples 2000`  
   `python main.py eval-adv-dev --model_dir runs/robust   --num_examples 2000`
8. **Run CheckList suite** (optional)  
   `python main.py run-checklist --model_dir runs/baseline`  
   `python main.py run-checklist --model_dir runs/robust`

---

## 6. Customization ideas

A few easy ways to extend this project:

- **Swap the base model**  
  Change `MODEL_NAME` in `config.py` (e.g., `bert-base-uncased`) and retrain.  
- **Use stronger distractors**  
  Edit `ADVERSARIAL_DISTRACTORS` to include more targeted or domain-specific sentences.  
- **Change train/adv proportions**  
  Right now, robust training simply concatenates original + adversarial data; you could downsample one or add an `adv_weight` scheme.  
- **Plug in a different dataset**  
  Replace `load_squad()` and the post-processing with equivalents for another QA dataset. The model/training logic should mostly carry over.  

---

## 7. Troubleshooting notes

- **AVX / TensorFlow warnings**  
  You may see messages like “To enable AVX2…”—they’re harmless here; the core training is in PyTorch.
- **Slow training**  
  On CPU this will be very slow. Use a GPU runtime (e.g., Colab T4/A100) if possible.
- **Different EM/F1 from the report**  
  Even with a fixed seed, small differences in library versions or hardware can move EM/F1 by a few tenths. That’s expected; the important thing is that the trends (baseline vs robust, clean vs adversarial) match.

If you’re using this for a course project or as a starting point for a product, this README is meant to be enough for someone else to rerun your experiments without digging through all the code.

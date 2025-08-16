# GPT2-UnHasher

*_A lightweight research project exploring whether a language model (gpt-2) can learn to map cryptographic hashes back to their original plaintext passwords._* 

_Since hashes are deterministic — the same input always produces the same output — this raises an open question: could a model eventually exploit subtle statistical patterns and begin to predict the plaintexts that produced given hashes?_  

_The goal here is not practical password recovery, but to test how far llms can generalize beyond natural language and model functions designed to be one-way._

---

## Table of Contents

- [Setup](#setup)
- [Quickstart](#quickstart)
- [Model Artifacts — `gpt2-hashmodel/`](#model-artifacts--gpt2-hashmodel)
- [Project Structure](#project-structure)
- [Data](#data)
- [Results](#results)
- [Script Reference (`src/gpt2_hasherv2.py`)](#script-reference-srcgpt2_hasherv2py)
- [.gitattributes (git lfs)](#gitattributes-git-lfs)
- [Tips / Troubleshooting](#tips--troubleshooting)
- [Responsible Use](#responsible-use)
- [License & Attribution](#license--attribution)

---

## Setup

1) create and activate a virtual environment (python ≥3.10 recommended).
    
    ```bash
    python -m venv .venv
    source .venv/bin/activate          # windows: .venv\Scripts\activate
    ```

2) install dependencies.
    
    ```bash
    pip install -r requirements.txt
    ```

3) (optional but recommended) enable git lfs for large model weights.
    
    ```bash
    git lfs install
    git lfs track "*.safetensors"
    git add .gitattributes
    git commit -m "track safetensors with git lfs"
    ```

---

## Quickstart

- **train + evaluate** (the script expects to be run from `src/` so that `Path.cwd().parent` resolves to the repo root):
    
    ```bash
    cd src
    python gpt2_hasherv2.py
    ```

- **what the script does**
  - loads model + tokenizer from: `../gpt2-hashmodel/`
  - streams training data from: `../data/shards/dev_shards.jsonl` (or the path you pass)
  - writes checkpoints + final weights to: `../gpt2-hashmodel/`
  - evaluates on: `../data/eval/dev_eval.yaml` (default), writing results to `../results/gpt/inferences_*.yaml`
  - in `__main__`, it runs 4 training/eval loops and overrides the training shard with:
    
    ```python
    'data/training/shards/shard_sample.jsonl'
    ```

  make sure your local data paths exist or adjust them when you run.

---

## Model Artifacts — `gpt2-hashmodel/`

this folder contains the fine-tuned model and tokenizer files used by `AutoModelForCausalLM` / `AutoTokenizer`:

- `model.safetensors` — final trained weights *(tracked via git lfs)*
- `config.json` — architecture/config
- `tokenizer.json` — serialized tokenizer *(or `vocab.json` + `merges.txt`)*
- `tokenizer_config.json` — tokenizer metadata
- `special_tokens_map.json` — special tokens (e.g. `<|endoftext|>`)
- `generation_config.json` — default generation settings
- `checkpoint-*/` — intermediate checkpoints *(kept locally; ignored by git)*

**load locally**:
    
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained("gpt2-hashmodel")
    model = AutoModelForCausalLM.from_pretrained("gpt2-hashmodel", torch_dtype="auto")
    model.eval()
    ```

---

## Project Structure

    ```text
    GPT2-UnHasher/
    ├─ gpt2-hashmodel/                # weights + tokenizer + configs (output_dir)
    ├─ results/
    │  └─ gpt/
    │     └─ inferences_*.yaml        # evaluation outputs (per run)
    ├─ data/
    │  ├─ shards/                     # training shards (jsonl; streamed)
    │  └─ eval/                       # eval yaml with input/output pairs
    ├─ src/
    │  ├─ gpt2_hasherv2.py            # main training/eval script
    │  └─ utils/
    │     └─ similarity.py            # char_similarity / levenshtein / jaccard
    ├─ requirements.txt
    ├─ .gitignore
    ├─ .gitattributes                 # lfs rules for *.safetensors
    └─ README.md
    ```

---

## Data

- training data is **not included**. the script streams jsonl shards.  
- expected default locations (relative to repo root):
  - training: `data/shards/dev_shards.jsonl` (or whatever you pass)
  - eval:     `data/eval/dev_eval.yaml`
- to use your own shard:
    
    ```bash
    cd src
    python gpt2_hasherv2.py  # edit __main__ or pass a different path into GPT2HashTuner(...)
    ```

**ignore raw datasets** in git. suggested entries are included below.

---

## Results

per-sample evaluation yaml format (written to `results/gpt/inferences_<STEM>_<ITER>.yaml`):

    ```yaml
    - hash: <hex hash>
      prediction: <model guess>
      truth: <ground truth>
      scores:
        aggregate: <mean of metrics>
        char_sim: <float>
        levenshtein: <float>
        jaccard: <float>
    ```

keep only representative runs (avoid huge dumps in the repo).

---

## Script Reference (`src/gpt2_hasherv2.py`)

- **paths** (computed with `Path.cwd().parent` so run from `src/`):
  - `_ROOT` = repo root
  - `_GPT2_MODEL` = `gpt2-hashmodel/` (save/load dir for model + tokenizer)
  - `_DEV_SHARD` = `data/shards/dev_shards.jsonl` (default train source)
  - `_EVAL_SRC`  = `data/eval/dev_eval.yaml` (default eval set)
  - `_EVAL_OUT`  = `results/gpt/inferences.yaml` (base; script appends stem/iter)

- **training**
  - masking: loss is computed only on the portion **after** the `Password:` marker
  - mixed precision: `fp16=True` by default
  - checkpoints: `save_steps`, `save_total_limit` (default 2–3 in your code)
  - no eval during training (`_EVAL_STRATEGY = 'no'`), evaluation happens post-training

- **evaluation**
  - generation defaults in script: `do_sample=True`, `temperature=0.01`, `top_k=1`, `max_new_tokens=24`, `pad_token_id=eos`
  - metrics: `char_similarity`, `levenshtein`, `jaccard` → `aggregate` is their mean
  - outputs appended as yaml to `results/gpt/`

---

## `.gitattributes` (git lfs)

    ```gitattributes
    *.safetensors filter=lfs diff=lfs merge=lfs -text
    ```

if you accidentally committed weights without lfs, migrate and force-push (_like I did_...):
    
    ```bash
    git lfs migrate import --include="*.safetensors"
    git push -f origin main
    ```

---

## Tips / Troubleshooting

- run from `src/` so `_ROOT = Path.cwd().parent` resolves correctly.
- if you add custom tokens, ensure `tokenizer.pad_token = tokenizer.eos_token` (the script does this).
- for cuda memory issues: lower `batch`, enable `gradient_checkpointing=True`, or reduce `max_length` in `_tokenize`.
- checkpoints can be large; keep them out of git and rely on `model.safetensors` as the final artifact.

---

## Responsible Use

this repository is for **academic research**. do **not** use it against real users, services, or sensitive systems. respect laws, tos, and ethical guidelines.

---

## License & Attribution

- base architecture/tokenizer: **gpt-2** (see original license).  
- fine-tuned weights and code here are released under the license specified in `LICENSE`.
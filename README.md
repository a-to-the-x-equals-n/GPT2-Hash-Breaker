# GPT2-Hasher

A lightweight experiment using GPT-2 to predict plaintext passwords from hashes.  
Built for academic research and curiosity.

---

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Change into the source directory:

```bash
cd src
```

3. Start training:

```bash
python gpt2_hasherv2.py
```

---

##  Model — `gpt2-hashmodel/`

This directory holds the fine-tuned GPT-2 model, checkpoints, and tokenizer data.

- `checkpoint-*/` — intermediate checkpoints saved during training
- `model.safetensors` — final trained model weights
- `config.json` — architecture settings
- `tokenizer.json` — full serialized tokenizer
- `tokenizer_config.json` — tokenizer metadata and behavior settings
- `vocab.json` + `merges.txt` — GPT-2's BPE vocabulary and merge rules
- `special_tokens_map.json` — defines special tokens like `<|endoftext|>`
- `generation_config.json` — default generation settings

You can load this model directly with `AutoTokenizer` and `AutoModelForCausalLM.from_pretrained('gpt2-hashmodel')`


## Structure

- `requirements.txt` — all needed Python packages
- `src/` — contains the model code and training script
- `data/` — training shards and evaluation files
- `results/` — output folder for inferences and metrics

---

## Notes

- This project is research-focused and uses synthetic hash/password pairs  
- Model is based on GPT-2 and fine-tuned for character-level prediction tasks

---

## Goals

- Test how well language models can generalize on non-natural language (e.g. hashes)
- Explore how far text generation can go in modeling one-way functions

---

For questions or contributions, reach out.
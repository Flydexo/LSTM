---
language:
  - fr
license: mit
tags:
  - lstm
  - language-model
  - poetry
  - french
  - lafontaine
  - text-generation
datasets:
  - flydexo/tinyfontaine
pipeline_tag: text-generation
---

# 🦊 Fontaine — LSTM French Fable Generator

> *"Rien ne sert de courir ; il faut partir à point."*
> — Jean de La Fontaine

**Fontaine** is a character-level language model that generates French poetry and fables in the style of Jean de La Fontaine (1621–1695), one of the great poets of the French classical period. It is built on a 3-layer LSTM with weight-tied embeddings and a custom BPE tokenizer trained on La Fontaine's *Fables*.

---

## Sample Output

```
<|titre|>le renard et le corbeau<|titre|>

Un renard, voyant du haut d'un chêne
Un corbeau qui tenait en son bec un fromage,
Lui tint à peu près ce langage :
— Hé ! bonjour, monsieur du Corbeau.
```

---

## Model Architecture

| Attribute | Value |
|---|---|
| Architecture | 3-layer LSTM |
| Hidden size | 256 |
| Tokenizer | BPE · 1 500 tokens |
| Context length | 72 tokens |
| Parameters | ~1.96M |
| Weight tying | Input embedding ↔ output projection |
| Regularization | Dropout (p = 0.4) |

The model is trained end-to-end with an AdamW optimizer and a OneCycleLR scheduler over 50 epochs.

---

## Training Data

Trained on [flydexo/tinyfontaine](https://huggingface.co/datasets/flydexo/tinyfontaine) — a curated dataset of fables and poems drawn from La Fontaine's complete *Fables* (Books I–XII). The dataset is split into training and validation sets and uses two special tokens to mark fable titles (`<|titre|>`) and section separators (`<|sep|>`).

---

## Quick Start

```python
from tokenizers import Tokenizer
from modeling_fontaine import FontaineLM
import torch

model = FontaineLM.from_pretrained("flydexo/fontaine", trust_remote_code=True)
tokenizer = Tokenizer.from_pretrained("flydexo/fontaine")
model.eval()

prompt = "<|titre|>le loup et l'agneau<|titre|>"
prompt_ids = tokenizer.encode(prompt).ids

generated_ids = model.generate_text(
    prompt_ids,
    max_new_tokens=300,
    temperature=0.8,
    device="cpu",
)
print(prompt, tokenizer.decode(generated_ids))
```

### Temperature guide

| Temperature | Output style |
|---|---|
| `0.5` | Conservative, repetitive |
| `0.8` | Balanced — recommended |
| `1.0` | Creative, occasionally surprising |
| `1.2+` | Very free-form |

---

## Special Tokens

| Token | ID | Role |
|---|---|---|
| `[UNK]` | `0` | Unknown token |
| `<\|titre\|>` | `1` | Opens **and** closes a fable title |
| `<\|sep\|>` | `2` | Section separator |

Wrap your prompt with `<|titre|>...<|titre|>` to condition the model on a fable title.

---

## Repo Structure

```
fontaine/
├── configuration_fontaine.py   # PretrainedConfig subclass
├── modeling_fontaine.py         # PreTrainedModel (LSTM + generate_text)
├── train.py                     # Full training script
├── config.json                  # Serialized hyperparameters
├── tokenizer.json               # HF tokenizers BPE file
└── README.md
```

---

## Training Details

```
Dataset  : flydexo/tinyfontaine
Optimizer: AdamW  (lr = 5e-3, weight_decay = 0.05)
Scheduler: OneCycleLR (pct_start = 0.3)
Epochs   : 50
Batch    : 64 streams × 72 tokens
Grad clip: 0.25
Device   : CUDA
```

---

## Limitations

- The model generates plausible French but does not guarantee semantic correctness.
- Training data is limited to a single author; the model will not generalize to other French poets.
- Not suitable for factual question answering, instruction following, or translation.

---

## Links

- [GitHub — training code & architecture](https://github.com/Flydexo/LSTM)
- [Dataset — flydexo/tinyfontaine](https://huggingface.co/datasets/flydexo/tinyfontaine)

## Citation

```bibtex
@misc{fontaine2025,
  author = {Flydexo},
  title  = {Fontaine: LSTM Language Model for La Fontaine-style French Fables},
  year   = {2025},
  url    = {https://huggingface.co/flydexo/fontaine}
}
```

---

## License

[MIT](LICENSE) — feel free to use, modify, and distribute.

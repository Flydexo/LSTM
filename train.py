"""Training script for FontaineLM — run this on a GPU machine, then push to HF Hub.

Re-run the script after a crash: if fontaine_checkpoint.pt exists, training is
skipped and the script goes straight to packaging + uploading.
"""

import os
import shutil
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from datasets import load_dataset
from huggingface_hub import HfApi
import matplotlib
matplotlib.use("Agg")  # headless GPU server
import matplotlib.pyplot as plt

from configuration_fontaine import FontaineConfig
from modeling_fontaine import FontaineLM

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEQ_LEN = 72
BATCH_SIZE = 64
N_HIDDEN = 256
N_LAYERS = 3
P_DROPOUT = 0.4
EPOCHS = 50
MAX_LR = 5e-3
WEIGHT_DECAY = 0.05
HF_REPO = "flydexo/fontaine"
CHECKPOINT = "fontaine_checkpoint.pt"

# ── Data ─────────────────────────────────────────────────────────────────────
ds = load_dataset("flydexo/tinyfontaine")
train_text = "\n".join(ds["train"]["text"])
valid_text = "\n".join(ds["validation"]["text"])

tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class FontaineDataset(torch.utils.data.IterableDataset):
    def __init__(self, text: str, seq_len: int, batch_size: int):
        super().__init__()
        ids = torch.tensor(tokenizer.encode(text).ids, device=device)
        n = len(ids) - 1
        tps = n // batch_size
        x = ids[: batch_size * tps].view(batch_size, -1)
        y = ids[1 : batch_size * tps + 1].view(batch_size, -1)
        self.batches = [
            (x[:, i : i + seq_len], y[:, i : i + seq_len])
            for i in range(0, x.shape[1] - seq_len + 1, seq_len)
            if x[:, i : i + seq_len].shape[1] == seq_len
        ]

    def __iter__(self):
        return iter(self.batches)


# ── Model ─────────────────────────────────────────────────────────────────────
config = FontaineConfig(
    vocab_size=vocab_size,
    n_hidden=N_HIDDEN,
    n_layers=N_LAYERS,
    p_dropout=P_DROPOUT,
    seq_len=SEQ_LEN,
)
model = FontaineLM(config).to(device)

# ── Skip training if a checkpoint already exists ──────────────────────────────
if os.path.exists(CHECKPOINT):
    print(f"Found {CHECKPOINT} — skipping training, loading weights for upload.")
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)  # fc.weight is tied, not in checkpoint
    model.fc.weight = model.embedding.weight
else:
    # ── Training ─────────────────────────────────────────────────────────────
    train_ds = FontaineDataset(train_text, SEQ_LEN, BATCH_SIZE)
    valid_ds = FontaineDataset(valid_text, SEQ_LEN, BATCH_SIZE)

    compiled = torch.compile(model)
    optimizer = torch.optim.AdamW(compiled.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_ds.batches),
        epochs=EPOCHS,
        pct_start=0.3,
    )

    train_losses, valid_losses = [], []

    for epoch in range(EPOCHS):
        compiled.train()
        total_loss, steps = 0.0, 0

        for x, y in train_ds:
            optimizer.zero_grad()
            out = compiled(x, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(compiled.parameters(), max_norm=0.25)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            steps += 1

        compiled.eval()
        model.reset_hidden()
        with torch.no_grad():
            vloss, v_steps = 0.0, 0
            for x, y in valid_ds:
                vloss += compiled(x, labels=y).loss.item()
                v_steps += 1
        model.reset_hidden()

        train_loss = total_loss / steps
        valid_loss = vloss / v_steps
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_losses, label="Train", color="#1f77b4", linewidth=2)
        plt.plot(range(1, epoch + 2), valid_losses, label="Valid", color="#ff7f0e", linewidth=2)
        plt.title(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Valid: {valid_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(f"loss_epoch_{epoch+1}.png", dpi=80, bbox_inches="tight")
        plt.close()
        print(f"Epoch {epoch+1}/{EPOCHS} | train={train_loss:.4f} | valid={valid_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

    # Save raw checkpoint — survives a crash in the upload step
    print(f"Saving checkpoint to {CHECKPOINT}...")
    torch.save(model.state_dict(), CHECKPOINT)

# ── Package & push ────────────────────────────────────────────────────────────
print("Packaging model...")
os.makedirs("./fontaine_weights", exist_ok=True)
model.save_pretrained("./fontaine_weights")
config.save_pretrained("./fontaine_weights")
shutil.copy("tokenizer.json", "./fontaine_weights/tokenizer.json")
for f in ("modeling_fontaine.py", "configuration_fontaine.py", "README.md"):
    shutil.copy(f, f"./fontaine_weights/{f}")

print(f"Pushing to {HF_REPO}...")
api = HfApi()
api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="./fontaine_weights",
    repo_id=HF_REPO,
    repo_type="model",
)
print("Done!")

#!/usr/bin/env python3
"""
BERT-Based Price Prediction Model Training - Amazon ML Challenge

Trains a BERT regression model on preprocessed product catalog text to predict prices.
Uses transfer learning from pretrained BERT with a regression head.

Dataset structure:
<root>/
  train/
    embeddings/
      content/             # *.txt with product text
    output/                # *.txt with target price (matching basename)
  val/                     # (optional)
    embeddings/
      content/
    output/

Example:
  python train_model.py \
    --root "training" \
    --model_name "bert-base-uncased" \
    --epochs 30 \
    --batch_size 16 \
    --lr 2e-5

Outputs:
  - best_model.pt (best checkpoint based on validation SMAPE)
  - train_manifest.csv, val_manifest.csv (debug manifests)
  - val_predictions.csv (validation predictions for analysis)
"""

import os
import re
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# ========================================================================
# HYPERPARAMETERS & CONFIGURATION
# ========================================================================
DEFAULT_ROOT = "/media/vibhor/hugedrive2/satvik/dd_training/aml/training"
DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_VALID_SIZE = 0.2
SEED = 42

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def set_seed(seed=SEED):
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smape(y_true, y_pred, eps=1e-8):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    Commonly used metric for price prediction competitions.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def first_float_in_text(s: str) -> Optional[float]:
    """
    Extract the first float-looking number in a string.
    Accepts integers or decimals, optional sign.
    """
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def read_text_file(p: Path) -> str:
    """Read text file with UTF-8 encoding and error handling."""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # Fallback for edge cases
        with p.open("rb") as f:
            return f.read().decode("utf-8", errors="ignore")

def read_label_file(p: Path) -> Optional[float]:
    """
    Try to read a label file that either contains just a number or
    has a number somewhere inside.
    """
    raw = read_text_file(p).strip()
    # Try direct float conversion first
    try:
        return float(raw)
    except Exception:
        pass
    # Otherwise extract first number from text
    val = first_float_in_text(raw)
    return val

def id_from_digits(name: str) -> Optional[str]:
    """
    Extract the first continuous sequence of digits from a basename.
    Return it as the ID string if found; else None.
    """
    m = re.search(r'\d+', name)
    return m.group(0) if m else None

def collect_split(root: Path,
                  split: str,
                  id_from_digits_only: bool = False) -> pd.DataFrame:
    """
    Build a manifest DataFrame with columns: ['id', 'text', 'price'] for a given split.
    If 'output' (labels) is missing, returns df with columns ['id', 'text'] only.

    Matching logic:
      - If id_from_digits_only=False (default): match on basename (without extension).
      - If id_from_digits_only=True: match on the first digits sequence in each basename.
    """
    emb_dir = root / split / "embeddings" / "content"
    out_dir = root / split / "output"

    if not emb_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {emb_dir}")

    has_labels = out_dir.is_dir()
    content_files = sorted([p for p in emb_dir.glob("**/*.txt")])

    # Build index maps (sample_id → file_path)
    text_map: Dict[str, Path] = {}
    for p in content_files:
        base = p.stem
        key = id_from_digits(base) if id_from_digits_only else base
        if key is not None:
            text_map[key] = p

    label_map: Dict[str, Path] = {}
    if has_labels:
        for p in sorted(out_dir.glob("**/*.txt")):
            base = p.stem
            key = id_from_digits(base) if id_from_digits_only else base
            if key is not None:
                label_map[key] = p

    rows = []
    misses = 0
    for key, tp in text_map.items():
        text = read_text_file(tp)
        row = {"id": key, "text": text}
        if has_labels:
            lp = label_map.get(key)
            if lp is None:
                misses += 1
            else:
                price = read_label_file(lp)
                if price is None:
                    misses += 1
                else:
                    row["price"] = float(price)
        rows.append(row)

    if has_labels and misses > 0:
        print(f"[{split}] WARNING: {misses} items had no usable label match.")

    df = pd.DataFrame(rows)
    # Ensure correct data types
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["price"])
        after = len(df)
        if after < before:
            print(f"[{split}] Dropped {before - after} rows with non-numeric price.")
    return df

# ========================================================================
# DATASET & MODEL
# ========================================================================

class PriceDataset(Dataset):
    """PyTorch Dataset for product text and price labels."""

    def __init__(self, df, tokenizer, is_train=True, max_len=128):
        self.is_train = is_train
        self.ids = df["id"].tolist()
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.targets = df["price"].values if (is_train and "price" in df.columns) else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Tokenize text and return tensors for model input."""
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
        # Some models don't use token_type_ids (e.g., RoBERTa)
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        else:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)
        item["ids"] = self.ids[idx]
        return item

class BertRegressor(nn.Module):
    """
    BERT-based regression model for price prediction.
    Uses pretrained BERT encoder + dropout + linear head.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        # Load pretrained BERT/transformer model
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        # Single-output regression head
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass: BERT encoding → pooling → dropout → linear head."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Use pooler output if available, otherwise do mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            x = outputs.pooler_output  # CLS token representation
        else:
            # Mean pooling over all tokens (masked)
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            x = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        x = self.dropout(x)
        out = self.head(x).squeeze(-1)
        return out

# ========================================================================
# TRAINING & EVALUATION
# ========================================================================

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Create linear warmup + decay scheduler for learning rate."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def train_one_epoch(model, loader, optimizer, device, scheduler=None, scaler=None):
    """Train for one epoch with optional mixed precision and LR scheduling."""
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        # Use mixed precision if GPU available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = nn.L1Loss()(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = nn.L1Loss()(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation set and compute metrics (L1, MAPE, SMAPE)."""
    model.eval()
    losses = []
    y_true_all = []
    y_pred_all = []
    for batch in tqdm(loader, desc="Validating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = nn.L1Loss()(outputs, targets)
        losses.append(loss.item())

        y_true_all.append(targets.detach().cpu().numpy())
        y_pred_all.append(outputs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    s = smape(y_true, y_pred)
    return float(np.mean(losses)), float(mape), float(s), y_true, y_pred

# ========================================================================
# MAIN TRAINING LOOP
# ========================================================================

def main():
    """Main training function: load data, train model, save best checkpoint."""
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Dataset root containing train/ and optional val/")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--valid_size", type=float, default=DEFAULT_VALID_SIZE, help="Used only if val/ split is missing")
    parser.add_argument("--id_from_digits", type=int, default=0, help="If 1, match files by first numeric ID in basename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)

    print(f"Root: {root}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")

    # Collect train split
    train_df = collect_split(root, "train", id_from_digits_only=bool(args.id_from_digits))
    if "price" not in train_df.columns:
        raise RuntimeError("Training split requires labels in train/output/*.txt")

    print(f"Train samples: {len(train_df)}")
    train_df.to_csv("train_manifest.csv", index=False)

    # Collect / decide validation
    val_dir = root / "val"
    if val_dir.exists():
        val_df = collect_split(root, "val", id_from_digits_only=bool(args.id_from_digits))
        if "price" not in val_df.columns:
            raise RuntimeError("Validation split 'val' exists but no labels found in val/output/*.txt")
        print(f"Val samples (from folder): {len(val_df)}")
    else:
        # auto split
        train_df, val_df = train_test_split(train_df, test_size=args.valid_size, random_state=SEED)
        print(f"Val samples (auto-split): {len(val_df)}")

    val_df.to_csv("val_manifest.csv", index=False)

    # Initialize tokenizer and create PyTorch datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ds_train = PriceDataset(train_df, tokenizer, is_train=True, max_len=args.max_len)
    ds_val = PriceDataset(val_df, tokenizer, is_train=True, max_len=args.max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model and training components
    model = BertRegressor(args.model_name).to(device)

    num_training_steps = args.epochs * len(dl_train)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_smape = float("inf")
    best_path = "best_model.pt"

    # Training loop: save best model based on validation SMAPE
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, dl_train, optimizer, device, scheduler, scaler)
        val_loss, val_mape, val_smape, y_true, y_pred = evaluate(model, dl_val, device)
        print(f"  train_L1: {tr_loss:.4f} | val_L1: {val_loss:.4f} | val_MAPE(%): {val_mape:.2f} | val_SMAPE(%): {val_smape:.2f}")

        if val_smape < best_smape:
            best_smape = val_smape
            torch.save(model.state_dict(), best_path)
            print(f"  New best SMAPE {best_smape:.2f}. Saved -> {best_path}")

    print("\nLoading best weights for final validation inference ...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Generate final validation predictions for analysis
    ids, preds, trues = [], [], []
    with torch.no_grad():
        for batch in tqdm(dl_val, desc="Val infer"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids).detach().cpu().numpy()
            preds.extend(outputs.tolist())
            trues.extend(batch["targets"].numpy().tolist())
            ids.extend(batch["ids"])

    pd.DataFrame({"id": ids, "price_true": trues, "price_pred": preds}).to_csv("val_predictions.csv", index=False)
    print("Saved val_predictions.csv")
    print("Done.")

if __name__ == "__main__":
    main()

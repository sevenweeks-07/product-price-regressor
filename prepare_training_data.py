#!/usr/bin/env python3
"""
Data Preprocessing for Amazon ML Challenge - Product Price Prediction

Prepares training data from raw CSV by:
  1. Cleaning catalog text (removing marketing fluff, keeping product details)
  2. Product-aware train/val splitting (prevents data leakage from product variants)
  3. Writing preprocessed data to folder structure for BERT training

Writes:
    training/
      ├── train/
      │   ├── embeddings/
      │   │   └── content/   (<sample_id>.txt)
      │   └── output/        (<sample_id>.txt)  # price only
      └── val/
          ├── embeddings/
          │   └── content/
          └── output/

CSV headers required: sample_id,catalog_content,price
(image_link may exist but is ignored)

Usage:
  python prepare_training_data.py \
      --csv path/to/train.csv \
      --out-dir training \
      --split 0.95 \
      --seed 42 \
      --workers 16 \
      --clean title_only      # minimal | smart | best | title_only (recommended)
      --infer-value-unit 0
"""

from __future__ import annotations
import os, re, csv, sys, argparse, random, html, unicodedata, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs): return x

# ========================================================================
# CSV READING (multiline-safe)
# ========================================================================

def read_rows(csv_path: str):
    """
    Read CSV file with proper handling of multiline fields.
    Returns list of dicts with sample_id and catalog_content.
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(
            f, delimiter=',', quotechar='"', doublequote=True, skipinitialspace=False
        )
        need = {"sample_id", "catalog_content"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain headers: {sorted(list(need))}. Got: {reader.fieldnames}")
        for r in reader:
            rows.append({
                "sample_id": str(r.get("sample_id", "")).strip(),
                "catalog_content": r.get("catalog_content", ""),
            })
    if not rows:
        raise ValueError("No rows read from CSV. Check file/path/encoding.")
    return rows

# ========================================================================
# TEXT CLEANING - Regex Patterns & Helper Functions
# ========================================================================

# Regex patterns to identify Value/Unit lines (keep these verbatim)
VALUE_LINE = re.compile(r'(?im)^\s*Value\s*:', re.M)
UNIT_LINE  = re.compile(r'(?im)^\s*Unit\s*:', re.M)

def strip_item_name_first_line(text: str) -> str:
    """Remove 'Item Name:' prefix from first line of product text."""
    if not isinstance(text, str): return ""
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = t.split('\n')
    if not lines: return ""
    lines[0] = re.sub(r'(?i)^\s*Item\s*Name\s*[:\-]\s*', '', lines[0]).strip()
    return '\n'.join(lines)

BULLET_LABEL_RE = re.compile(r'(?im)^\s*Bullet\s*Point\s*\d+\s*:\s*')
PROD_DESC_LABEL_RE = re.compile(r'(?im)^\s*Product\s*Description\s*:\s*')
HTML_TAG_RE  = re.compile(r"<[^>]+>")
PRICE_RE     = re.compile(r"(?i)(?:price|mrp|rs\.?|₹|inr|usd|\$)\s*[\d,]+(?:\.\d+)?")
SKU_NOISE_RE = re.compile(r"(?i)\b(?:SKU|ASIN|UPC|EAN|MPN|ZIN)\s*[:#]?\s*[A-Z0-9\-]+\b")
CODE_NOISE_RE= re.compile(r"(?i)\b(?:model|item|code|ref|id)\s*[:#]?\s*[A-Z0-9\-]+\b")
WS_RE        = re.compile(r"\s+")
DUP_PUNCT_RE = re.compile(r"([!?,.;:])\1{1,}")

def _norm_unit(u: str) -> str:
    """Normalize unit strings (e.g., 'ounces' → 'oz', 'grams' → 'g')."""
    u = (u or "").strip().lower().replace("ounces", "oz").replace("ounce", "oz")
    u = u.replace("pounds", "lb").replace("pound", "lb")
    u = u.replace("grams", "g").replace("gram", "g")
    u = u.replace("milliliters", "ml").replace("milliliter", "ml")
    u = u.replace("liters", "l").replace("litres", "l").replace("liter", "l").replace("litre", "l")
    u = re.sub(r"\s+", " ", u)
    if "fl" in u and "oz" in u: return "fl oz"
    return u

INLINE_QTY_ALL = re.compile(
    r'(\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)\s*(fl\s*oz|oz|ml|l|g|kg|lb|pound|pounds|gram|grams|'
    r'liter|litre|liters|litres|ounce|ounces)\b',
    re.IGNORECASE
)

def _unicode_fraction_to_decimal(s: str) -> str:
    """Convert unicode fraction characters (½, ¼, etc.) to decimal strings."""
    table = {
        "½":"0.5", "¼":"0.25", "¾":"0.75",
        "⅓":"0.333", "⅔":"0.667", "⅛":"0.125", "⅜":"0.375", "⅝":"0.625", "⅞":"0.875"
    }
    return "".join(table.get(ch, ch) for ch in s)

def _normalize_whitespace(s: str) -> str:
    """Normalize whitespace and remove duplicate punctuation."""
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width
    s = unicodedata.normalize("NFKC", s)
    s = WS_RE.sub(" ", s)
    s = DUP_PUNCT_RE.sub(r"\1", s)
    return s.strip()

def _largest_inline_qty(text: str):
    """Return (value_string, normalized_unit) from largest numeric token we can find (string, not float)."""
    best = None
    for num, unit in INLINE_QTY_ALL.findall(text):
        # clean fraction like '1/2' or '2,5'
        n = num.replace(",", ".")
        if "/" in n:
            try:
                a,b = n.split("/")
                val = float(a) / float(b)
            except Exception:
                continue
        else:
            try:
                val = float(n)
            except Exception:
                continue
        u = _norm_unit(unit)
        # prefer the largest numeric (no cross-unit conversion here)
        if (best is None) or (val > best[0]):
            best = (val, u, num)
    if best is None:
        return None, None
    _, u, raw = best
    return raw, u

# ========================================================================
# PRODUCT GROUPING - Hash products to prevent train/val data leakage
# ========================================================================

def product_hash(text: str) -> str:
    """
    Hash product title (first line, normalized) to group product variants.
    Removes digits so different sizes/packs get the same hash.
    """
    if not isinstance(text, str) or not text.strip():
        return hashlib.md5(b"").hexdigest()
    
    # Get first line (title)
    first_line = text.split('\n')[0][:120].strip().lower()
    
    # Remove digits to group variants (e.g., "Coke 12oz" and "Coke 24oz" → same hash)
    normalized = re.sub(r'\d+', '0', first_line)
    
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

# ========================================================================
# TEXT CLEANING MODES - Four strategies with increasing aggressiveness
# ========================================================================

def clean_minimal(text: str) -> str:
    """Minimal cleaning: just remove bullet labels, keep everything else."""
    if not isinstance(text, str): return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ln in t.split("\n"):
        if not ln.strip(): continue
        if VALUE_LINE.match(ln) or UNIT_LINE.match(ln):
            out.append(ln.strip()); continue
        ln = BULLET_LABEL_RE.sub("", ln).strip()
        out.append(ln)
    return "\n".join(out).strip()

def clean_smart(text: str) -> str:
    """Smart cleaning: keep only lines with quantity/size info or short titles."""
    if not isinstance(text, str): return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for raw in t.split("\n"):
        ln = raw.strip()
        if not ln: continue
        if VALUE_LINE.match(ln) or UNIT_LINE.match(ln):
            out.append(ln); continue
        ln = BULLET_LABEL_RE.sub("", ln).strip()
        ln = PROD_DESC_LABEL_RE.sub("", ln).strip()
        # Only keep lines with quantity/size info
        if any(kw in ln.lower() for kw in ['oz', 'ml', 'g', 'kg', 'lb', 'pack', 'count', 'pcs', 'ct']):
            out.append(ln)
        elif len(ln) <= 100:  # Keep short lines (likely titles)
            out.append(ln)
    out = [re.sub(r'(?i)\bZIN\s*:\s*\d+\b', '', ln).strip() for ln in out if ln.strip()]
    return "\n".join(out).strip()

def clean_best(text: str, infer_value_unit: bool = False) -> str:
    """
    Aggressive cleaner:
      - Remove ALL marketing fluff
      - Keep ONLY: title + quantities/sizes + Value/Unit
      - Much simpler than original
    """
    if not isinstance(text, str): return ""

    # Basic normalization
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = strip_item_name_first_line(t)
    t = html.unescape(t)
    t = _unicode_fraction_to_decimal(t)
    t = HTML_TAG_RE.sub(" ", t)

    lines = []
    for raw in t.split("\n"):
        ln = raw.strip()
        if not ln:
            continue
        
        # Always keep Value/Unit
        if VALUE_LINE.match(ln) or UNIT_LINE.match(ln):
            lines.append(ln)
            continue
        
        # Clean the line
        ln = BULLET_LABEL_RE.sub("", ln).strip()
        ln = PROD_DESC_LABEL_RE.sub("", ln).strip()
        ln = PRICE_RE.sub(" ", ln)
        ln = SKU_NOISE_RE.sub(" ", ln)
        ln = CODE_NOISE_RE.sub(" ", ln)
        ln = _normalize_whitespace(ln)
        
        # Skip if empty after cleaning
        if not ln or len(ln) < 5:
            continue
        
        # ONLY keep lines with size/quantity info OR short title-like text
        has_size_info = bool(re.search(r'\b\d+\s*(?:oz|ml|g|kg|lb|pack|count|pcs|ct|x)\b', ln, re.I))
        is_short_title = len(ln) <= 100 and not any(fluff in ln.lower() for fluff in [
            'perfect for', 'great for', 'ideal', 'premium', 'quality', 
            'satisfaction', 'guarantee', 'trust', 'best', 'founded'
        ])
        
        if has_size_info or is_short_title:
            lines.append(ln)

    # Infer Value/Unit if requested
    if infer_value_unit:
        text_no_labels = "\n".join([x for x in lines if not (VALUE_LINE.match(x) or UNIT_LINE.match(x))])
        have_value = any(VALUE_LINE.match(x) for x in lines)
        have_unit  = any(UNIT_LINE.match(x) for x in lines)
        
        if not have_value or not have_unit:
            raw_qty, unit = _largest_inline_qty(text_no_labels)
            if raw_qty and unit:
                lines.append(f"Value: {raw_qty}")
                lines.append(f"Unit: {_norm_unit(unit)}")

    return "\n".join(lines).strip()

def clean_title_only(text: str) -> str:
    """
    SIMPLEST & MOST EFFECTIVE: Keep only title (first line) + Value/Unit
    Removes all marketing fluff automatically.
    """
    if not isinstance(text, str): return ""
    
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    output = []
    
    # Keep first line as title (most important)
    if lines:
        title = strip_item_name_first_line(lines[0])
        title = html.unescape(title)
        title = _unicode_fraction_to_decimal(title)
        title = re.sub(r'\s+', ' ', title).strip()
        if title:
            output.append(title)
    
    # Keep Value/Unit lines
    for ln in lines:
        ln_stripped = ln.strip()
        if VALUE_LINE.match(ln_stripped) or UNIT_LINE.match(ln_stripped):
            output.append(ln_stripped)
    
    return '\n'.join(output)

# ========================================================================
# I/O HELPERS
# ========================================================================

def make_dirs(root: str):
    """Create directory structure for train/val splits."""
    paths = {
        "train_content": os.path.join(root, "train", "embeddings", "content"),
        "train_output":  os.path.join(root, "train", "output"),
        "val_content":   os.path.join(root, "val",   "embeddings", "content"),
        "val_output":    os.path.join(root, "val",   "output"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def write_text(path: str, text: str):
    """Write text to file with UTF-8 encoding."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text if isinstance(text, str) else "")


# ========================================================================
# DATA PROCESSING
# ========================================================================

def process_record(rec: dict, content_dir: str, out_dir: str, clean_mode: str, infer_value_unit: bool):
    """Process a single record: clean text and write to disk."""
    sid = rec["sample_id"]
    content_path = os.path.join(content_dir, f"{sid}.txt")

    txt = rec.get("catalog_content", "")

    if clean_mode == "minimal":
        cleaned = clean_minimal(txt)
    elif clean_mode == "smart":
        cleaned = clean_smart(txt)
    elif clean_mode == "best":
        cleaned = clean_best(txt, infer_value_unit=infer_value_unit)
    else:  # title_only
        cleaned = clean_title_only(txt)

    write_text(content_path, cleaned)
    return {"sample_id": sid, "ok": True}

# ========================================================================
# MAIN ENTRY POINT
# ========================================================================

def main():
    """Main function: parse arguments, prepare data, and create train/val splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to train.csv (multiline-safe)")
    ap.add_argument("--out-dir", default="training", help="Output root folder")
    ap.add_argument("--split", type=float, default=1.0, help="Train fraction (0-1). Default 0.95")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--workers", type=int, default=16, help="Thread workers for I/O")
    ap.add_argument("--clean", choices=["minimal", "smart", "best", "title_only"], default="title_only",
                    help="Text cleaning mode (title_only recommended)")
    ap.add_argument("--infer-value-unit", type=int, default=0,
                    help="If 1, append inferred Value/Unit when missing")
    args = ap.parse_args()

    print("Reading CSV...")
    rows = read_rows(args.csv)

    # Dedupe sample_ids (keep last)
    uniq = {}
    for r in rows: 
        uniq[r["sample_id"]] = r
    rows = list(uniq.values())
    print(f"Total unique samples: {len(rows)}")

    # ===== PRODUCT-AWARE SPLIT (KEY FIX!) =====
    print("Grouping by product (to avoid train/val overlap)...")
    product_groups = defaultdict(list)
    for r in rows:
        h = product_hash(r.get('catalog_content', ''))
        product_groups[h].append(r)
    
    print(f"Found {len(product_groups)} unique product groups")
    
    # Split GROUPS, not individual rows
    rng = random.Random(args.seed)
    group_keys = list(product_groups.keys())
    rng.shuffle(group_keys)
    cut = int(len(group_keys) * args.split)
    
    train_rows = []
    val_rows = []
    for k in group_keys[:cut]:
        train_rows.extend(product_groups[k])
    for k in group_keys[cut:]:
        val_rows.extend(product_groups[k])
    
    print(f"Split: {len(train_rows)} train rows | {len(val_rows)} val rows")
    print(f"  ({len(group_keys[:cut])} train groups | {len(group_keys[cut:])} val groups)")

    # dirs
    paths = make_dirs(args.out_dir)

    # process
    def run_batch(batch_rows, content_dir, out_dir, label):
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(process_record, r, content_dir, out_dir, args.clean, bool(args.infer_value_unit))
                       for r in batch_rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
                _ = fut.result()

    run_batch(train_rows, paths["train_content"], paths["train_output"], "train")
    run_batch(val_rows,   paths["val_content"],   paths["val_output"],   "val")

    # summary
    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60)
    print(f"Train: {len(train_rows)} samples")
    print(f"Val:   {len(val_rows)} samples")
    print(f"Root:  {os.path.abspath(args.out_dir)}")
    print(f"Cleaning mode: {args.clean}")
    print(f"Product groups used to prevent overlap: {len(product_groups)}")
    print("\nStructure:")
    print(f"  {args.out_dir}/train/embeddings/content/<sample_id>.txt")
    print(f"  {args.out_dir}/train/output/<sample_id>.txt")
    print(f"  {args.out_dir}/val/embeddings/content/<sample_id>.txt")
    print(f"  {args.out_dir}/val/output/<sample_id>.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
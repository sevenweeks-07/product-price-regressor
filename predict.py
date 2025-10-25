#!/usr/bin/env python3
"""
BERT Price Inference - Direct CSV Input
Matches the training script preprocessing (no text cleaning).
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from contextlib import nullcontext
from pathlib import Path

# =============== MODEL DEFINITION ===============
class BertRegressor(nn.Module):
    """BERT-based regression model for price prediction."""
    
    def __init__(self, model_name: str, dropout: float = 0.1):
        """
        Initialize the BERT regression model.
        
        Args:
            model_name: Name of pretrained model (e.g., 'bert-base-uncased')
            dropout: Dropout rate for regularization
        """
        # Call parent class constructor
        super().__init__()
        
        # Load pretrained BERT model as backbone encoder
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size from BERT's configuration
        hidden = self.backbone.config.hidden_size
        
        # Add dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Add linear layer to output a single regression value
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Mask for padded tokens [batch_size, seq_len]
            token_type_ids: Segment IDs for BERT [batch_size, seq_len]
        
        Returns:
            Predicted prices as 1D tensor [batch_size]
        """
        # Pass inputs through BERT backbone
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Check if model has pooler_output (CLS token representation)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            # Use pre-computed pooler output
            x = out.pooler_output
        else:
            # Manually compute mean pooling over token embeddings
            last = out.last_hidden_state  # [batch, seq_len, hidden_size]
            
            # Expand attention mask to match embedding dimensions
            mask = attention_mask.unsqueeze(-1).expand(last.size()).float()
            
            # Compute mean: sum(embeddings * mask) / sum(mask)
            x = torch.sum(last * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Pass through linear head and squeeze to 1D
        return self.head(x).squeeze(-1)


# =============== DATASET ===============
class TextDataset(Dataset):
    """Dataset class for text data with tokenization."""
    
    def __init__(self, ids, texts, tokenizer, max_len):
        """
        Initialize dataset.
        
        Args:
            ids: List of sample IDs (strings)
            texts: List of text content (strings)
            tokenizer: HuggingFace tokenizer instance
            max_len: Maximum sequence length for tokenization
        """
        # Store sample IDs
        self.ids = ids
        
        # Store text content
        self.texts = texts
        
        # Store tokenizer reference
        self.tk = tokenizer
        
        # Store max sequence length
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples in dataset."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Get a single sample by index.
        
        Args:
            idx: Index of sample to retrieve
        
        Returns:
            Dictionary containing tokenized inputs and sample ID
        """
        # Get text at this index
        t = self.texts[idx]
        
        # Tokenize the text with padding and truncation
        enc = self.tk(
            t,                           # Input text
            max_length=self.max_len,     # Truncate to max length
            truncation=True,             # Enable truncation
            padding="max_length",        # Pad to max length
            return_tensors="pt"          # Return PyTorch tensors
        )
        
        # Get token_type_ids if available (for BERT segment embeddings)
        tok = enc.get("token_type_ids")
        
        # If token_type_ids not present, create zeros tensor
        if tok is None:
            tok = torch.zeros_like(enc["input_ids"])
        
        # Return dictionary with all required tensors
        return {
            "input_ids": enc["input_ids"].squeeze(0),        # Remove batch dimension
            "attention_mask": enc["attention_mask"].squeeze(0),  # Remove batch dimension
            "token_type_ids": tok.squeeze(0),                # Remove batch dimension
            "id": self.ids[idx],                             # Sample ID as string
        }


# =============== INFERENCE FUNCTION ===============
def run_inference(
    input_csv,
    output_csv,
    model_path,
    model_name="bert-base-uncased",
    max_len=128,
    batch_size=64,
    device_str=None,
    clip_min=None,
):
    """
    Run price prediction inference on CSV data.
    
    Args:
        input_csv: Path to input CSV file with columns: sample_id, catalog_content
        output_csv: Path to save predictions CSV
        model_path: Path to saved model checkpoint (.pt file)
        model_name: Name of pretrained model used during training
        max_len: Maximum sequence length (should match training)
        batch_size: Batch size for inference
        device_str: Device to use ('cuda' or 'cpu', None=auto)
        clip_min: Optional minimum value to clip predictions
    """
    # Determine device (use GPU if available, otherwise CPU)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load the input CSV file
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Validate required columns exist in CSV
    required_cols = {"sample_id", "catalog_content"}
    if not required_cols.issubset(df.columns):
        # Raise error if required columns missing
        raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")
    
    # Create a copy to avoid modifying original dataframe
    df = df.copy()
    
    # Convert sample_id to string type and rename to 'id'
    df["id"] = df["sample_id"].astype(str)
    
    # NO PREPROCESSING - Use raw catalog_content as-is (matching training script)
    # Convert to string and fill any NaN values with empty string
    df["text"] = df["catalog_content"].fillna("").astype(str)
    
    print(f"Sample text (first 200 chars): {df['text'].iloc[0][:200]}...")

    # Load the tokenizer for the specified model
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize the model architecture
    print(f"Initializing model: {model_name}")
    model = BertRegressor(model_name)

    # Load trained weights from checkpoint file
    print(f"Loading model weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Load state dict into model (allow missing/unexpected keys)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Warn about any missing keys
    if missing_keys:
        print(f"[WARNING] Missing keys in checkpoint: {missing_keys}")
    
    # Warn about any unexpected keys
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys}")

    # Move model to target device and set to evaluation mode
    model.to(device)
    model.eval()

    # Create dataset from dataframe
    dataset = TextDataset(
        ids=df["id"].tolist(),      # List of sample IDs
        texts=df["text"].tolist(),  # List of text content
        tokenizer=tokenizer,        # Tokenizer instance
        max_len=max_len             # Max sequence length
    )
    
    # Create dataloader for batched inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=False,          # Don't shuffle (preserve order)
        num_workers=2,          # Number of worker processes for data loading
        pin_memory=True         # Pin memory for faster GPU transfer
    )

    # Helper function to get appropriate autocast context
    def get_amp_context():
        """Return autocast context for mixed precision if on CUDA."""
        if device.type == "cuda":
            # Use automatic mixed precision on GPU
            return torch.amp.autocast(device_type="cuda")
        else:
            # No autocast on CPU
            return nullcontext()

    # Initialize lists to collect predictions and IDs
    all_predictions = []
    all_ids = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Iterate through batches with progress bar
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move input_ids to device (non-blocking for async GPU transfer)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            
            # Move attention_mask to device
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            # Move token_type_ids to device
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)

            # Safety check: if tokenizer returned 3D tensor, squeeze middle dimension
            if token_type_ids.ndim == 3 and token_type_ids.size(1) == 1:
                token_type_ids = token_type_ids.squeeze(1)

            # Forward pass through model with mixed precision
            with get_amp_context():
                predictions = model(input_ids, attention_mask, token_type_ids)
            
            # Move predictions to CPU and convert to numpy
            all_predictions.append(predictions.detach().cpu().numpy())
            
            # Collect batch IDs (already on CPU)
            all_ids.extend(batch["id"])

    # Concatenate all batch predictions into single array
    predictions = np.concatenate(all_predictions).astype(np.float64)
    
    # Optional: clip predictions to minimum value if specified
    if clip_min is not None:
        print(f"Clipping predictions to minimum value: {clip_min}")
        predictions = np.clip(predictions, clip_min, None)

    # Create output dataframe with sample IDs and predictions
    output_df = pd.DataFrame({
        "id": all_ids,                    # Sample IDs
        "predicted_price": predictions    # Predicted prices
    })
    
    # Create output directory if it doesn't exist
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Save predictions to CSV file
    output_df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"{'='*60}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Mean predicted price: ${predictions.mean():.2f}")
    print(f"Min predicted price: ${predictions.min():.2f}")
    print(f"Max predicted price: ${predictions.max():.2f}")
    print(f"Saved predictions to: {output_csv}")


# =============== MAIN ===============
def main():
    """Main function to parse arguments and run inference."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run BERT price inference on CSV file"
    )
    
    # Required arguments
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV with columns: sample_id, catalog_content"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output_csv",
        default="predictions.csv",
        help="Path to save predictions CSV (default: predictions.csv)"
    )
    
    # Optional model arguments
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        help="Pretrained model name (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cuda' or 'cpu' (default: auto-detect)"
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=None,
        help="Optional minimum value for predictions (e.g., 0.05)"
    )
    
    # Parse command line arguments
    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*60)
    print("BERT Price Prediction - Inference")
    print("="*60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Model path: {args.model_path}")
    print(f"Model name: {args.model_name}")
    print(f"Max length: {args.max_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output CSV: {args.output_csv}")
    print("="*60 + "\n")
    
    # Run inference with parsed arguments
    run_inference(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_path=args.model_path,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device_str=args.device,
        clip_min=args.clip_min,
    )


if __name__ == "__main__":
    # Execute main function
    main()
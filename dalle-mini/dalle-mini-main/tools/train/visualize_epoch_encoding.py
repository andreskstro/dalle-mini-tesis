#!/usr/bin/env python3
"""
Visualize saved epoch encoding
==============================
- Loads an encoding .npy saved by train.py (shape: (1,16,16), dtype=int32)
- Prints basic stats and the prompt used (if .txt exists)
- Saves a simple 16x16 color grid PNG to inspect token layout (NOT decoded image)

Usage:
  python tools/train/visualize_epoch_encoding.py \
    --encoding /path/to/output_dir/epoch_encodings/encoding_epoch_1.npy \
    --out_png /path/to/output_dir/epoch_encodings/encoding_epoch_1.png

Optional:
  --cmap viridis   # any matplotlib colormap
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", required=True, help="Path to encoding_epoch_*.npy")
    parser.add_argument("--out_png", required=False, help="Path to save visualization PNG")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--print_tokens", action="store_true", help="Print the 16x16 token matrix to stdout")
    parser.add_argument("--save_txt", action="store_true", help="Save tokens as a plain text matrix next to the PNG")
    parser.add_argument("--save_csv", action="store_true", help="Save tokens as CSV next to the PNG")
    args = parser.parse_args()

    enc_path = os.path.abspath(args.encoding)
    if not os.path.exists(enc_path):
        raise FileNotFoundError(enc_path)

    # Try to read companion prompt
    prompt_txt = None
    base, _ = os.path.splitext(enc_path)
    txt_path = base + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt_txt = f.read().strip()

    arr = np.load(enc_path)
    print(f"Loaded: {enc_path}")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    flat = arr.reshape(-1)
    print(f"Stats: min={flat.min()}, max={flat.max()}, mean={flat.mean():.3f}")
    if prompt_txt:
        print(f"Prompt: {prompt_txt}")

    # Expect shape (1,16,16) but handle any 2D/3D gracefully
    if arr.ndim == 3 and arr.shape[0] == 1:
        img = arr[0]
    elif arr.ndim == 2:
        img = arr
    else:
        # Fallback to best-effort: take last 256 tokens and reshape 16x16
        tokens = flat[-256:]
        img = tokens.reshape(16, 16)
        print("Note: reshaped last 256 tokens to 16x16 for visualization")

    # Optionally print/save raw token IDs
    if args.print_tokens:
        print("\nTokens (16x16):")
        with np.printoptions(linewidth=200, threshold=256):
            print(img.astype(int))

    base_out = args.out_png or base + ".png"
    out_dir = os.path.dirname(base_out)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_txt:
        txt_tokens_path = base + "_tokens.txt"
        np.savetxt(txt_tokens_path, img.astype(int), fmt="%d")
        print(f"Saved tokens TXT: {txt_tokens_path}")
    if args.save_csv:
        csv_tokens_path = base + "_tokens.csv"
        np.savetxt(csv_tokens_path, img.astype(int), fmt="%d", delimiter=",")
        print(f"Saved tokens CSV: {csv_tokens_path}")

    # Plot as colored grid (token IDs as values)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(img, cmap=args.cmap, interpolation="nearest")
    ax.set_title("Encoding tokens (16x16)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_png = base_out
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved visualization: {out_png}")


if __name__ == "__main__":
    main()

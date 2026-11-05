import os
import sys
import math
import argparse
from glob import glob

from transformers import AutoTokenizer, BartTokenizerFast
import pandas as pd


def find_repo_root() -> str:
    # This file: <root>/TesisBack/tools/offline/count_tokens_dalle.py
    here = os.path.abspath(__file__)
    tesis_back = os.path.dirname(os.path.dirname(os.path.dirname(here)))  # .../TesisBack
    root = os.path.dirname(tesis_back)  # .../<root>
    return root


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def main():
    parser = argparse.ArgumentParser(description="Count token lengths with dalle-mini tokenizer")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Directory with .txt prompts (defaults to <repo_root>/tesisfront/train_data)",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Filename pattern to include (glob)",
    )
    parser.add_argument(
        "--parquet_dir",
        default=None,
        help="Directory with .parquet files containing prompts",
    )
    parser.add_argument(
        "--parquet_pattern",
        default="*.parquet",
        help="Filename pattern for parquet files (glob)",
    )
    parser.add_argument(
        "--text_column",
        default=None,
        help="Column name in parquet that contains the prompt text",
    )
    parser.add_argument(
        "--hf_model",
        default="dalle-mini/dalle-mini",
        help="Hugging Face tokenizer repo to use",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust custom code from the HF repo (required for dalle-mini tokenizer)",
    )
    args = parser.parse_args()

    root = find_repo_root()
    dataset_dir = args.dataset_dir or os.path.join(root, "tesisfront", "train_data")

    print(f"Loading tokenizer: {args.hf_model} ...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        print(f"[INFO] AutoTokenizer failed ({e}). Falling back to BartTokenizerFast...", file=sys.stderr)
        tokenizer = BartTokenizerFast.from_pretrained(args.hf_model, use_fast=True)

    txt_paths = []
    if dataset_dir and os.path.isdir(dataset_dir):
        txt_paths = sorted(glob(os.path.join(dataset_dir, args.pattern)))
    parquet_paths = []
    if args.parquet_dir and os.path.isdir(args.parquet_dir):
        parquet_paths = sorted(glob(os.path.join(args.parquet_dir, args.parquet_pattern)))
        if parquet_paths and not args.text_column:
            print("text_column is required when using --parquet_dir", file=sys.stderr)
            sys.exit(2)
    if not txt_paths and not parquet_paths:
        print("No input files found. Provide --dataset_dir or --parquet_dir with valid files.", file=sys.stderr)
        sys.exit(0)

    lengths = []
    max_len = -1
    max_files = []

    counter = 0
    for i, p in enumerate(txt_paths, 1):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue

        # Encode with special tokens as used typically for conditioning
        ids = tokenizer.encode(text, add_special_tokens=True)
        L = len(ids)
        lengths.append(L)
        if L > max_len:
            max_len = L
            max_files = [p]
        elif L == max_len:
            max_files.append(p)
        counter += 1
        if counter % 5000 == 0:
            print(f"Processed {counter} items... current max={max_len}")

    for pq_path in parquet_paths:
        try:
            df = pd.read_parquet(pq_path, columns=[args.text_column] if args.text_column else None)
        except Exception as e:
            print(f"[WARN] Failed to read parquet {pq_path}: {e}")
            continue
        if args.text_column not in df.columns:
            print(f"[WARN] Column '{args.text_column}' not in {pq_path} columns {list(df.columns)}")
            continue
        for idx, text in enumerate(df[args.text_column].astype(str).tolist()):
            ids = tokenizer.encode(text.strip(), add_special_tokens=True)
            L = len(ids)
            lengths.append(L)
            tag = f"{pq_path}#row={idx}"
            if L > max_len:
                max_len = L
                max_files = [tag]
            elif L == max_len:
                max_files.append(tag)
            counter += 1
            if counter % 5000 == 0:
                print(f"Processed {counter} items... current max={max_len}")

    lengths.sort()
    n = len(lengths)
    mean = sum(lengths) / n if n else 0
    p95 = percentile(lengths, 95)
    p99 = percentile(lengths, 99)

    print("\n==== Token Stats (dalle-mini/dalle-mini) ====")
    print(f"Items counted: {n}")
    print(f"Max tokens:   {max_len}")
    print(f"Mean tokens:  {mean:.2f}")
    print(f"P95 tokens:   {p95:.2f}")
    print(f"P99 tokens:   {p99:.2f}")
    print("Files with max:")
    for p in max_files[:20]:
        try:
            if "#row=" in p:
                file_part, row_part = p.split("#row=")
                rel = os.path.relpath(file_part, root)
                print(f" - {rel}#row={row_part}")
            else:
                print(f" - {os.path.relpath(p, root)}")
        except Exception:
            print(f" - {p}")
    if len(max_files) > 20:
        print(f" ... and {len(max_files) - 20} more")


if __name__ == "__main__":
    main()

import os
import sys
import math
import argparse
from glob import glob

# Intentar importar pandas
try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' no está instalado. Ejecuta: pip install pandas", file=sys.stderr)
    sys.exit(1)

from transformers import AutoTokenizer, BartTokenizerFast


def find_repo_root() -> str:
    here = os.path.abspath(__file__)
    # Ajuste simple para intentar encontrar la raíz, aunque no es crítico para este script
    return os.path.dirname(here)


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
    
    # Argumentos para .TXT
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Directory with .txt prompts",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Filename pattern to include for txt files (glob)",
    )

    # Argumentos para .PARQUET
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

    # NUEVO: Argumentos para .CSV
    parser.add_argument(
        "--csv_dir",
        default=None,
        help="Directory with .csv files containing prompts",
    )
    parser.add_argument(
        "--csv_pattern",
        default="*.csv",
        help="Filename pattern for csv files (glob)",
    )

    # Columna común para CSV y Parquet
    parser.add_argument(
        "--text_column",
        default="caption",  # Puse 'caption' por defecto para facilitarte la vida
        help="Column name in csv/parquet that contains the prompt text",
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

    # 1. Buscar archivos TXT
    txt_paths = []
    if args.dataset_dir and os.path.isdir(args.dataset_dir):
        txt_paths = sorted(glob(os.path.join(args.dataset_dir, args.pattern)))

    # 2. Buscar archivos PARQUET
    parquet_paths = []
    if args.parquet_dir and os.path.isdir(args.parquet_dir):
        parquet_paths = sorted(glob(os.path.join(args.parquet_dir, args.parquet_pattern)))

    # 3. Buscar archivos CSV (NUEVO)
    csv_paths = []
    if args.csv_dir and os.path.isdir(args.csv_dir):
        csv_paths = sorted(glob(os.path.join(args.csv_dir, args.csv_pattern)))

    if not txt_paths and not parquet_paths and not csv_paths:
        print("No input files found. Provide --dataset_dir, --parquet_dir, or --csv_dir.", file=sys.stderr)
        sys.exit(0)

    lengths = []
    max_len = -1
    max_files = []
    counter = 0

    # Procesar TXT
    for p in txt_paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
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
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")

    # Procesar PARQUET
    for pq_path in parquet_paths:
        if not args.text_column:
            print(f"[ERROR] --text_column required for parquet file: {pq_path}", file=sys.stderr)
            continue
        try:
            df = pd.read_parquet(pq_path, columns=[args.text_column])
            for idx, text in enumerate(df[args.text_column].astype(str).tolist()):
                ids = tokenizer.encode(text.strip(), add_special_tokens=True)
                L = len(ids)
                lengths.append(L)
                tag = f"{os.path.basename(pq_path)}#row={idx}"
                if L > max_len:
                    max_len = L
                    max_files = [tag]
                elif L == max_len:
                    max_files.append(tag)
                counter += 1
                if counter % 5000 == 0:
                    print(f"Processed {counter} items... current max={max_len}")
        except Exception as e:
            print(f"[WARN] Failed to read parquet {pq_path}: {e}")

    # Procesar CSV (NUEVO)
    for csv_path in csv_paths:
        if not args.text_column:
            print(f"[ERROR] --text_column required for csv file: {csv_path}", file=sys.stderr)
            continue
        try:
            # Leemos el CSV
            df = pd.read_csv(csv_path)
            if args.text_column not in df.columns:
                print(f"[WARN] Column '{args.text_column}' not found in {csv_path}. Columns are: {list(df.columns)}")
                continue
            
            # Iteramos filas
            for idx, text in enumerate(df[args.text_column].astype(str).tolist()):
                ids = tokenizer.encode(text.strip(), add_special_tokens=True)
                L = len(ids)
                lengths.append(L)
                tag = f"{os.path.basename(csv_path)}#row={idx}"
                
                if L > max_len:
                    max_len = L
                    max_files = [tag]
                elif L == max_len:
                    max_files.append(tag)
                
                counter += 1
                if counter % 5000 == 0:
                    print(f"Processed {counter} items... current max={max_len}")

        except Exception as e:
            print(f"[WARN] Failed to read csv {csv_path}: {e}")

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
        print(f" - {p}")
    if len(max_files) > 20:
        print(f" ... and {len(max_files) - 20} more")


if __name__ == "__main__":
    main()
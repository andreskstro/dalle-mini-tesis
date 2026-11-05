import argparse
import os
import io
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from PIL import Image

# JAX/Flax VQGAN
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from vqgan_jax.modeling_flax_vqgan import VQModel

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

_vqgan = None
_vq_params = None


def ensure_model():
    global _vqgan, _vq_params
    if _vqgan is None or _vq_params is None:
        vqgan, vq_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )
        _vqgan = vqgan
        _vq_params = replicate(vq_params)
    return _vqgan, _vq_params


def decode_encoding_to_pil(encoding: List[int]) -> Image.Image:
    vqgan, vq_params = ensure_model()
    if not isinstance(encoding, (list, tuple)) or len(encoding) == 0:
        raise ValueError("encoding is empty or not a list")
    if isinstance(encoding[0], (list, tuple, np.ndarray)):
        tokens = np.asarray(encoding, dtype=np.int32).ravel()
    else:
        tokens = np.asarray(encoding, dtype=np.int32)
    L = int(tokens.shape[0])
    # Expect a flat sequence of length image_length (e.g., 256)
    seq = jnp.array(tokens)[None, None, :]  # [n_devices, per_device_batch, L] with n_devices=1

    @jax.pmap
    def p_decode(batch_tokens, params):
        return vqgan.decode_code(batch_tokens, params=params)

    decoded = p_decode(seq, vq_params)
    img = np.array(decoded)[0, 0]  # [H,W,3] in [-1,1]
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def gather_parquet_paths(in_dir: str) -> List[str]:
    paths = [str(p) for p in Path(in_dir).rglob("*.parquet")]
    if not paths:
        raise FileNotFoundError(f"No .parquet files found under: {in_dir}")
    return sorted(paths)


def load_index_or_scan(paths: List[str], wanted: set | None, max_needed: int, match_mode: str = "filename") -> Dict[str, Tuple[str, int]]:
    index: Dict[str, Tuple[str, int]] = {}
    remaining = None if wanted is None else set(wanted)
    for p in paths:
        # Read in chunks to avoid large memory usage
        try:
            df = pd.read_parquet(p, columns=["filename"])  # only filename first
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}")
            continue
        if wanted is None:
            # Take first rows if sampling will happen later
            for i, fn in enumerate(df["filename"].astype(str)):
                if fn not in index:
                    index[fn] = (p, i)
                if len(index) >= max_needed:
                    return index
        else:
            col = df["filename"].astype(str)
            if match_mode == "filename":
                mask = col.isin(remaining)
                if mask.any():
                    hit_positions = df.index[mask].tolist()
                    for row_idx in hit_positions:
                        fn = str(col.iloc[row_idx])
                        index[fn] = (p, int(row_idx))
                        if fn in remaining:
                            remaining.remove(fn)
                        if not remaining:
                            return index
            elif match_mode == "stem":
                stems = col.map(lambda s: Path(s).stem)
                stem_to_rows: Dict[str, List[int]] = {}
                for ridx, st in zip(df.index.tolist(), stems.tolist()):
                    stem_to_rows.setdefault(st, []).append(int(ridx))
                hits = [st for st in remaining if st in stem_to_rows]
                for st in hits:
                    ridx = stem_to_rows[st][0]
                    fn = str(col.iloc[ridx])
                    index[fn] = (p, int(ridx))
                    remaining.remove(st)
                    if not remaining:
                        return index
        if wanted is not None and not remaining:
            break
    return index


def export_items(index: Dict[str, Tuple[str, int]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fn, (pq, row_idx) in index.items():
        try:
            df = pd.read_parquet(pq, columns=["filename", "caption", "encoding"])
            row = df.iloc[row_idx]
            caption = str(row["caption"]) if not pd.isna(row["caption"]) else ""
            enc = row["encoding"]
            if not isinstance(enc, (list, tuple)):
                enc = list(enc)
            img = decode_encoding_to_pil(enc)
            # Build safe file stem
            # Keep only last component of filename as stem
            stem = Path(fn).name
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
            img_path = os.path.join(out_dir, f"{stem}.png")
            txt_path = os.path.join(out_dir, f"{stem}.txt")
            img.save(img_path, format="PNG")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption.strip())
            print(f"Wrote: {img_path} and {txt_path}")
        except Exception as e:
            print(f"[error] Failed to export {fn} from {pq} at row {row_idx}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing parquet files")
    ap.add_argument("--out_dir", required=True, help="Output directory for PNG+TXT")
    ap.add_argument("--filenames", nargs="*", help="Specific filenames (as stored in parquet column 'filename')")
    ap.add_argument("--match_mode", choices=["filename","stem"], default="filename", help="How to match provided identifiers: exact filename or stem (without folder/extension)")
    ap.add_argument("--n", type=int, default=5, help="Number of items to export if filenames not provided")
    args = ap.parse_args()

    paths = gather_parquet_paths(args.in_dir)

    if args.filenames:
        wanted = set(map(str, args.filenames))
        index = load_index_or_scan(paths, wanted=wanted, max_needed=len(wanted), match_mode=args.match_mode)
        # For stem mode, compute missing relative to wanted set of stems
        if args.match_mode == "stem":
            found_stems = {Path(k).stem for k in index.keys()}
            missing = wanted - found_stems
        else:
            missing = wanted - set(index.keys())
        if missing:
            print(f"[warn] Missing filenames: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
    else:
        # Build a small index by scanning sequentially and sampling
        rapid_index = load_index_or_scan(paths, wanted=None, max_needed=max(args.n, 5), match_mode="filename")
        keys = list(rapid_index.keys())
        random.shuffle(keys)
        keys = keys[: args.n]
        index = {k: rapid_index[k] for k in keys}

    if not index:
        print("No items found to export.")
        return

    export_items(index, args.out_dir)


if __name__ == "__main__":
    main()

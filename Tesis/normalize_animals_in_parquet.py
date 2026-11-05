import re
import pandas as pd
from pathlib import PurePosixPath, PureWindowsPath
import argparse
import generate_pokemon_captions as gpc

ANIMAL_RX = gpc._animal_regex_en()

def pokemon_from_filename(path: str) -> str:
    if not path:
        return ""
    # If path has folders, assume first component is the species folder (properly cased)
    if "/" in path:
        parts = PurePosixPath(path).parts
        return parts[0] if parts else ""
    if "\\" in path:
        parts = PureWindowsPath(path).parts
        return parts[0] if parts else ""
    # Otherwise, derive from stem before underscore: e.g., "abra_0002.jpg" -> "Abra"
    stem = PurePosixPath(path).stem
    species = stem.split("_")[0] if "_" in stem else stem
    # Basic casing: capitalize first letter, keep rest lower
    if not species:
        return ""
    return species[:1].upper() + species[1:].lower()

def fix_caption(caption: str, pokemon: str, filename: str | None = None) -> str:
    if not caption or not pokemon:
        return caption or ""
    s = caption
    # Replace any occurrence of the filename stem (e.g., abra_0002) by the Pokémon name
    if filename:
        stem = PurePosixPath(filename).stem
        if stem:
            s = re.sub(rf"\b{re.escape(stem)}\b", pokemon, s, flags=re.IGNORECASE)
    # Replace detected animals and 'toy' with the proper Pokémon name (preserve casing)
    s = ANIMAL_RX.sub(pokemon, s)
    s = re.sub(r"\btoy\b", pokemon, s, flags=re.IGNORECASE)
    s = re.sub(r"\babra\b", pokemon, s, flags=re.IGNORECASE)
    if pokemon[:1].lower() in "aeiou":
        s = re.sub(r"\ba\s+(" + re.escape(pokemon) + r")\b", r"an \1", s, flags=re.IGNORECASE)
    else:
        s = re.sub(r"\ban\s+(" + re.escape(pokemon) + r")\b", r"a \1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def process_parquet(in_path: str, out_path: str, filename_col: str = "filename", caption_col: str = "caption", encoding_col: str = "encoding") -> None:
    df = pd.read_parquet(in_path)
    required = {filename_col, caption_col, encoding_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Input parquet must contain columns: {sorted(required)}")
    fns = df[filename_col].astype(str)
    pokemons = fns.map(pokemon_from_filename)
    df[caption_col] = [fix_caption(c, p, fn) for c, p, fn in zip(df[caption_col].astype(str), pokemons, fns)]
    df.to_parquet(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--caption_col", default="caption")
    ap.add_argument("--encoding_col", default="encoding")
    args = ap.parse_args()
    process_parquet(args.in_path, args.out_path, args.filename_col, args.caption_col, args.encoding_col)

if __name__ == "__main__":
    main()

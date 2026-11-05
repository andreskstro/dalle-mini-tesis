import os
import sys
import argparse
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

from PIL import Image
import imagehash
from tqdm import tqdm

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ALLOWED_EXTS


def bucket_key(h: imagehash.ImageHash, bits: int = 16) -> str:
    # Use the first 'bits' to bucketize, each hex char = 4 bits
    return str(h)[: bits // 4]


def walk_images(dataset: str) -> List[Tuple[str, str]]:
    """Return list of (abs_path, rel_path_from_dataset)."""
    items: List[Tuple[str, str]] = []
    for root, _, files in os.walk(dataset):
        for f in sorted(files):
            if is_image(f):
                abs_p = os.path.join(root, f)
                rel_p = os.path.relpath(abs_p, dataset).replace('\\', '/')
                items.append((abs_p, rel_p))
    return items


def ensure_unique(dst_path: str) -> str:
    base, ext = os.path.splitext(dst_path)
    k = 1
    out = dst_path
    while os.path.exists(out):
        out = f"{base}__dup{k}{ext}"
        k += 1
    return out


def dedupe(dataset: str, scope: str, threshold: int, dry_run: bool, action: str, quarantine_dir: str, bucket_bits: int = 16) -> Tuple[int, List[Tuple[str, str, int]]]:
    images = walk_images(dataset)
    if not images:
        print("No images found.")
        return 0, []

    duplicates: List[Tuple[str, str, int]] = []  # (dup_rel, orig_rel, dist)

    if scope == 'global':
        reps: DefaultDict[str, List[Tuple[imagehash.ImageHash, str, str]]] = defaultdict(list)  # bucket -> [(hash, abs, rel)]
        for abs_p, rel_p in tqdm(images, desc='Scanning', unit='img'):
            try:
                with Image.open(abs_p) as img:
                    h = imagehash.phash(img)
            except Exception as e:
                print(f"WARN: failed to hash {rel_p}: {e}")
                continue
            bkey = bucket_key(h, bits=bucket_bits)
            matched = False
            for h_ref, abs_ref, rel_ref in reps[bkey]:
                dist = (h - h_ref)
                if dist <= threshold:
                    duplicates.append((rel_p, rel_ref, dist))
                    matched = True
                    break
            if not matched:
                reps[bkey].append((h, abs_p, rel_p))
    else:  # per_folder
        folders: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)
        for abs_p, rel_p in images:
            folder = rel_p.split('/', 1)[0]
            folders[folder].append((abs_p, rel_p))
        for folder, items in tqdm(folders.items(), desc='Folders', unit='folder'):
            reps: DefaultDict[str, List[Tuple[imagehash.ImageHash, str, str]]] = defaultdict(list)
            for abs_p, rel_p in items:
                try:
                    with Image.open(abs_p) as img:
                        h = imagehash.phash(img)
                except Exception as e:
                    print(f"WARN: failed to hash {rel_p}: {e}")
                    continue
                bkey = bucket_key(h, bits=bucket_bits)
                matched = False
                for h_ref, abs_ref, rel_ref in reps[bkey]:
                    dist = (h - h_ref)
                    if dist <= threshold:
                        duplicates.append((rel_p, rel_ref, dist))
                        matched = True
                        break
                if not matched:
                    reps[bkey].append((h, abs_p, rel_p))

    print(f"Found {len(duplicates)} duplicates (threshold={threshold}).")

    if dry_run:
        # Show a small preview
        for dup_rel, orig_rel, dist in duplicates[:20]:
            print(f"DUP ~{dist}: {dup_rel} -> {orig_rel}")
        if len(duplicates) > 20:
            print("...")
        print("Dry-run mode: no files were changed.")
        return len(duplicates), duplicates

    # Apply action
    if action not in { 'delete', 'move' }:
        raise ValueError("action must be 'delete' or 'move'")

    moved = 0
    deleted = 0
    if action == 'move':
        os.makedirs(quarantine_dir, exist_ok=True)

    for dup_rel, orig_rel, dist in tqdm(duplicates, desc='Applying', unit='file'):
        src_abs = os.path.join(dataset, dup_rel)
        if not os.path.exists(src_abs):
            continue
        if action == 'delete':
            try:
                os.remove(src_abs)
                deleted += 1
            except Exception as e:
                print(f"WARN: failed to delete {dup_rel}: {e}")
        else:  # move
            # mirror folder inside quarantine
            folder, fname = os.path.split(dup_rel)
            dst_folder = os.path.join(quarantine_dir, folder)
            os.makedirs(dst_folder, exist_ok=True)
            dst_abs = os.path.join(dst_folder, fname)
            dst_abs = ensure_unique(dst_abs)
            try:
                os.replace(src_abs, dst_abs)
                moved += 1
            except Exception as e:
                print(f"WARN: failed to move {dup_rel}: {e}")

    if action == 'delete':
        print(f"Done. Deleted {deleted} duplicates.")
    else:
        print(f"Done. Moved {moved} duplicates to: {quarantine_dir}")

    return len(duplicates), duplicates


def main():
    parser = argparse.ArgumentParser(description='Find and handle duplicate/similar images using pHash, supporting folder-per-Pokémon structure.')
    parser.add_argument('--dataset', default='dataset', help='Dataset root with subfolders per Pokémon')
    parser.add_argument('--scope', choices=['global', 'per_folder'], default='global', help='Deduplicate across whole dataset or within each folder')
    parser.add_argument('--threshold', type=int, default=3, help='pHash distance threshold (lower = stricter)')
    parser.add_argument('--bucket_bits', type=int, default=16, help='Bits used to bucket hashes (tradeoff speed/recall)')
    parser.add_argument('--dry_run', action='store_true', help='Preview duplicates without changing files')
    parser.add_argument('--action', choices=['delete', 'move'], default='move', help='What to do with detected duplicates')
    parser.add_argument('--quarantine', default='_duplicates', help='Folder (inside dataset) to move duplicates when action=move')

    args = parser.parse_args()

    dataset = args.dataset
    if not os.path.isdir(dataset):
        print(f"❌ Dataset directory not found: {dataset}")
        sys.exit(1)

    quarantine_dir = args.quarantine
    if not os.path.isabs(quarantine_dir):
        quarantine_dir = os.path.join(dataset, quarantine_dir)

    dedupe(dataset, args.scope, args.threshold, args.dry_run, args.action, quarantine_dir, bucket_bits=args.bucket_bits)


if __name__ == '__main__':
    main()

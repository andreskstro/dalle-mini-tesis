import os
import argparse
from typing import List, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_subfolders(path: str) -> List[str]:
    return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]


def list_images(path: str) -> List[str]:
    return [f for f in sorted(os.listdir(path)) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def list_txts(path: str) -> List[str]:
    return [f for f in sorted(os.listdir(path)) if f.lower().endswith(".txt")]


def ensure_unique(dst_path: str) -> str:
    """If dst_path exists, append incremental suffix until it's unique."""
    base, ext = os.path.splitext(dst_path)
    k = 1
    unique_path = dst_path
    while os.path.exists(unique_path):
        unique_path = f"{base}__{k}{ext}"
        k += 1
    return unique_path


def main():
    parser = argparse.ArgumentParser(description="Delete caption .txt files and rename images to a readable pattern per folder.")
    parser.add_argument("--dataset", default="dataset", help="Root dataset directory with subfolders per Pokémon")
    parser.add_argument("--delete_txt", action="store_true", help="Delete .txt caption files found inside each subfolder")
    parser.add_argument("--dry_run", action="store_true", help="Show planned actions without applying changes")
    parser.add_argument("--lowercase", action="store_true", help="Use lowercase folder name in filenames (default)")
    parser.add_argument("--uppercase", action="store_true", help="Use capitalized folder name in filenames")
    parser.add_argument("--width", type=int, default=4, help="Zero-pad width for index (default: 4)")
    parser.add_argument("--start", type=int, default=1, help="Starting index for numbering (default: 1)")
    parser.add_argument("--sep", default="_", help="Separator between name and index (default: '_')")

    args = parser.parse_args()

    dataset_dir = args.dataset
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # name casing strategy
    def name_for_folder(folder: str) -> str:
        if args.uppercase:
            return folder
        # default lowercase unless uppercase explicitly requested
        return folder.lower()

    folders = list_subfolders(dataset_dir)
    total_imgs = 0
    planned_moves: List[Tuple[str, str]] = []
    planned_deletes: List[str] = []

    for folder in folders:
        fpath = os.path.join(dataset_dir, folder)
        # delete txts
        if args.delete_txt:
            for txt in list_txts(fpath):
                planned_deletes.append(os.path.join(fpath, txt))
        # rename images
        images = list_images(fpath)
        total_imgs += len(images)
        idx = args.start
        name_root = name_for_folder(folder)
        for img in images:
            src = os.path.join(fpath, img)
            ext = os.path.splitext(img)[1].lower()
            target_name = f"{name_root}{args.sep}{str(idx).zfill(args.width)}{ext}"
            dst = os.path.join(fpath, target_name)
            if os.path.abspath(src) == os.path.abspath(dst):
                # already well-named for this index, just increment
                idx += 1
                continue
            # Avoid collision with existing files
            if os.path.exists(dst):
                dst = ensure_unique(dst)
            planned_moves.append((src, dst))
            idx += 1

    # Report plan
    print(f"Folders: {len(folders)} | Images: {total_imgs}")
    print(f"Planned deletes: {len(planned_deletes)} | Planned renames: {len(planned_moves)}")

    if args.dry_run:
        # Show only first few actions to avoid huge output
        preview = 10
        print("\nPreview deletes:")
        for p in planned_deletes[:preview]:
            print(f"DELETE: {p}")
        if len(planned_deletes) > preview:
            print("...")
        print("\nPreview renames:")
        for s, d in planned_moves[:preview]:
            print(f"RENAME: {s} -> {d}")
        if len(planned_moves) > preview:
            print("...")
        print("\nDry-run complete. No changes applied.")
        return

    # Apply deletes
    deleted = 0
    for p in planned_deletes:
        try:
            os.remove(p)
            deleted += 1
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"WARN: Failed to delete {p}: {e}")

    # Apply renames
    renamed = 0
    for src, dst in planned_moves:
        try:
            os.rename(src, dst)
            renamed += 1
        except FileNotFoundError:
            # Source may have been moved already due to a collision or manual changes
            continue
        except PermissionError as e:
            print(f"WARN: Permission error renaming {src} -> {dst}: {e}")
        except Exception as e:
            print(f"WARN: Failed to rename {src} -> {dst}: {e}")

    print(f"Done. Deleted {deleted} .txt files and renamed {renamed} images.")


if __name__ == "__main__":
    main()

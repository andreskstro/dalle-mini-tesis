"""
Preview de un shard .tar de WebDataset.

- Lista los primeros N miembros (nombre y tamaño)
- Muestra conteo por extensión
- Extrae y guarda algunas imágenes de muestra (jpg/png/webp) en una carpeta destino

Uso:
  python tools/dataset/preview_tar.py --tar backend/dalle-mini-main/data/synth1m/data-000000.tar \
      --max-list 50 --sample-images 3 --out-dir backend/dalle-mini-main/data/synth1m/preview

Notas:
- Si PIL (Pillow) está instalado, intentará validar/apertura de imágenes; si no, solo extrae bytes.
- Solo soporta rutas locales .tar.
"""
from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path
import tarfile

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def is_image_member(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar", required=True, help="Ruta local al archivo .tar a previsualizar")
    p.add_argument("--max-list", type=int, default=50, help="Número máximo de miembros a listar")
    p.add_argument("--sample-images", type=int, default=3, help="Número de imágenes a extraer como muestra")
    p.add_argument("--out-dir", default=None, help="Directorio donde guardar imágenes de muestra")
    args = p.parse_args()

    tar_path = Path(args.tar)
    assert tar_path.exists(), f"No existe: {tar_path}"

    print(f"Abriendo: {tar_path}")
    with tarfile.open(tar_path, mode="r:*") as tar:
        members = tar.getmembers()
        total = len(members)
        print(f"Total de miembros: {total}")

        # Conteo por extensión
        ext_count = collections.Counter(Path(m.name).suffix.lower() for m in members)
        print("\nConteo por extensión (top 20):")
        for ext, cnt in ext_count.most_common(20):
            print(f"  {ext or '(sin ext)'}: {cnt}")

        # Lista primeros N
        print(f"\nPrimeros {min(args.max_list, total)} miembros:")
        for m in members[: args.max_list]:
            print(f"  {m.name}  |  {human_bytes(m.size)}")

        # Extraer imágenes de muestra
        img_members = [m for m in members if m.isfile() and is_image_member(m.name)]
        if not img_members:
            print("\nNo se encontraron imágenes (.jpg/.jpeg/.png/.webp) en el tar.")
            return

        n = min(args.sample_images, len(img_members))
        if n <= 0:
            return

        out_dir = Path(args.out_dir) if args.out_dir else tar_path.parent / "preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExtrayendo {n} imágenes de muestra en: {out_dir}")

        for m in img_members[:n]:
            extracted = tar.extractfile(m)
            if extracted is None:
                continue
            out_path = out_dir / (Path(m.name).name)
            with open(out_path, "wb") as f:
                f.write(extracted.read())
            print(f"  -> {out_path}")
            if PIL_AVAILABLE:
                try:
                    Image.open(out_path).verify()  # valida estructura
                    print("     (OK) imagen verificada por PIL")
                except Exception as e:
                    print(f"     (WARN) PIL no pudo verificar la imagen: {e}")


if __name__ == "__main__":
    main()

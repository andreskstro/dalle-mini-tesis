# -*- coding: utf-8 -*-
# Valida y decodifica una fila de un parquet/CSV con 'encoding' de VQGAN

import os
import ast
import math
import numpy as np
import glob
import csv

from collections import Counter

# ==== CONFIG ====
FILE_PATH = "/mnt/c/Users/Andrés Castro/Documents/GitHub/TesisBackend/backend/dalle-mini-main/tools/dataset/encoded_data/newtrain++++"
FILE_TYPE = "parquet"                      # "parquet" o "csv"
ROW_INDEX = 0                              # <- Índice de la fila a inspeccionar
SAVE_PNG = True
OUT_PNG = f"recon_row{ROW_INDEX}.png"
SKIP_DECODE = False                        # True = solo validar, no decodificar
TOKENS_HEATMAP_PNG = f"tokens_row{ROW_INDEX}.png"

# VQGAN unificado
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# =================

def _resolve_parquet_file(path: str) -> str:
    """Devuelve un archivo .parquet. Si es un directorio, toma el primero encontrado."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No se encontraron .parquet en: {path}")
        return files[0]
    if os.path.isfile(path) and path.lower().endswith(".parquet"):
        return path
    raise FileNotFoundError(f"Ruta no válida para parquet: {path}")

def _load_encoding_from_parquet(path: str, row_index: int):
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise ImportError("Falta pyarrow para leer Parquet. Instala con: pip install pyarrow") from e

    file_path = _resolve_parquet_file(path)
    pf = pq.ParquetFile(file_path)

    # Validaciones de esquema
    names = pf.schema_arrow.names
    if "encoding" not in names:
        raise KeyError("La tabla no contiene columna 'encoding'")

    total_rows = pf.metadata.num_rows
    if total_rows == 0:
        raise ValueError("El parquet no contiene filas")
    if row_index < 0 or row_index >= total_rows:
        raise IndexError(f"ROW_INDEX fuera de rango (0..{total_rows-1})")

    # Iterar por lotes de 1 fila para no cargar todo y evitar dataset/pandas
    current = 0
    for batch in pf.iter_batches(columns=["encoding"], batch_size=1):
        if current == row_index:
            # batch es RecordBatch con una columna
            arr = batch.column(0)
            # tomar primer elemento del batch
            value = arr[0].as_py()
            return value
        current += 1

    # Si no se retornó nada (no debería ocurrir)
    raise IndexError("ROW_INDEX no encontrado al iterar el Parquet")

def _load_encoding_from_csv(path: str, row_index: int):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No se encontraron .csv en: {path}")
        path = files[0]
    if not (os.path.isfile(path) and path.lower().endswith(".csv")):
        raise FileNotFoundError(f"Ruta no válida para csv: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV vacío")
        if "encoding" not in header:
            raise KeyError("La tabla no contiene columna 'encoding'")
        enc_idx = header.index("encoding")
        for i, row in enumerate(reader):
            if i == row_index:
                return row[enc_idx]
        raise IndexError("ROW_INDEX fuera de rango en CSV")

def _load_encoding(path, ftype, row_index):
    if ftype == "parquet":
        return _load_encoding_from_parquet(path, row_index)
    elif ftype == "csv":
        return _load_encoding_from_csv(path, row_index)
    else:
        raise ValueError("FILE_TYPE debe ser 'parquet' o 'csv'")

def _parse_encoding(enc):
    # enc puede ser: list[int], np.ndarray, string tipo "[1,2,...]"
    if isinstance(enc, (list, np.ndarray)):
        arr = np.array(enc, dtype=np.int32).ravel()
        return arr
    if isinstance(enc, str):
        s = enc.strip()
        # quitar posibles comillas externas
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]
        # intentar literal_eval de la lista
        try:
            parsed = ast.literal_eval(s)
            arr = np.array(parsed, dtype=np.int32).ravel()
            return arr
        except Exception as e:
            raise ValueError(f"No pude parsear 'encoding' desde string: {e}")
    raise ValueError(f"Tipo de 'encoding' no soportado: {type(enc)}")

def _print_stats(codes):
    print(f"- len(codes): {len(codes)}")
    print(f"- rango: min={codes.min()} max={codes.max()}")
    uniq = np.unique(codes)
    print(f"- únicos: {len(uniq)}")
    cnt = Counter(codes.tolist())
    top = cnt.most_common(10)
    print("- top 10 tokens y conteos:")
    for tok, c in top:
        print(f"  token {tok}: {c}")

def _reshape_codes(codes):
    # Ideal: 256 -> 16x16
    n = len(codes)
    if n == 256:
        return codes.reshape(1, 16, 16), (16, 16), ""
    # Fallback: intentar cuadrado perfecto
    s = int(math.isqrt(n))
    if s * s == n:
        warn = f"Advertencia: longitud {n} no es 256; usando {s}x{s} por ser cuadrado perfecto."
        return codes.reshape(1, s, s), (s, s), warn
    raise ValueError(f"Longitud de codes={n} no es 256 ni cuadrado perfecto.")

def _decode_and_save(z_idx_1hw):
    # Import aquí para evitar cargar JAX si solo quieres validar estadísticas
    from vqgan_jax.modeling_flax_vqgan import VQModel
    import jax.numpy as jnp

    vqgan, vq_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT, _do_init=False
    )

    # Preparar variantes de forma/dtype de índices por compatibilidad
    import jax.numpy as jnp  # ya importado arriba, redundante pero seguro en scopes
    z_variants = []
    base = np.array(z_idx_1hw)
    flat = base.reshape(1, -1)
    shapes = [
        base.shape,            # (1,16,16)
        (1, flat.shape[1]),    # (1,256)
        (base.shape[1], base.shape[2]),  # (16,16)
        (flat.shape[1],),      # (256,)
    ]
    dtypes = [jnp.int32, jnp.int64]
    built = set()
    for shp in shapes:
        for dt in dtypes:
            try:
                arr = jnp.array(base.reshape(shp), dtype=dt)
                key = (arr.shape, int(arr.dtype==jnp.int64))
                if key not in built:
                    z_variants.append(arr)
                    built.add(key)
            except Exception:
                continue

    # Probar varias rutas de decodificación para compatibilidad entre versiones
    recon = None
    errors = []

    # 1) decode_code con variantes (kw y posicional)
    if hasattr(vqgan, "decode_code"):
        for z_idx in z_variants:
            try:
                out = vqgan.decode_code(z_idx, params=vq_params)
                recon = out[0] if isinstance(out, (tuple, list)) else out
                print(f"Decodificación OK: decode_code(params=), shape={tuple(z_idx.shape)}, dtype={z_idx.dtype}")
                break
            except Exception as e:
                errors.append(f"decode_code(params=) shape={tuple(z_idx.shape)} dtype={z_idx.dtype}: {e}")
        if recon is None:
            for z_idx in z_variants:
                try:
                    out = vqgan.decode_code(z_idx, vq_params)
                    recon = out[0] if isinstance(out, (tuple, list)) else out
                    print(f"Decodificación OK: decode_code(positional), shape={tuple(z_idx.shape)}, dtype={z_idx.dtype}")
                    break
                except Exception as e:
                    errors.append(f"decode_code(positional) shape={tuple(z_idx.shape)} dtype={z_idx.dtype}: {e}")

    # 2) apply(..., method=decode_code)
    if recon is None and hasattr(vqgan, "decode_code"):
        try:
            recon = vqgan.apply({"params": vq_params}, z_idx, method=vqgan.decode_code)
            print("Decodificación OK con método: apply(method=decode_code)")
        except Exception as e:
            errors.append(f"apply(method=decode_code): {e}")
            recon = None

    # 3) decode(z_idx, params=...)  (nota: decode espera latentes cuantizados, no índices)
    if recon is None and hasattr(vqgan, "decode"):
        try:
            recon = vqgan.decode(z_idx, params=vq_params)
            print("Decodificación OK con método: decode()")
        except Exception as e:
            errors.append(f"decode(params=): {e}")
            recon = None

    # 4) apply(..., method=decode)
    if recon is None and hasattr(vqgan, "decode"):
        try:
            recon = vqgan.apply({"params": vq_params}, z_idx, method=vqgan.decode)
            print("Decodificación OK con método: apply(method=decode)")
        except Exception as e:
            errors.append(f"apply(method=decode): {e}")
            recon = None

    if recon is None:
        raise RuntimeError("No se pudo decodificar. Errores: " + "; ".join(errors))

    # Algunas versiones pueden devolver tuplas (recon, extra)
    if isinstance(recon, (tuple, list)):
        recon = recon[0]

    img = np.array(recon)[0]                   # HWC, float en [-1,1]
    img = np.clip((img + 1.0) / 2.0, 0.0, 1.0) # [0,1]
    if SAVE_PNG:
        try:
            from PIL import Image
            im8 = (img * 255).astype(np.uint8)
            Image.fromarray(im8).save(OUT_PNG)
            print(f"Imagen guardada en: {os.path.abspath(OUT_PNG)}")
        except Exception as e:
            print(f"No pude guardar PNG con PIL: {e}")
    return img

def main():
    print(f"Cargando: {FILE_PATH}")
    value = _load_encoding(FILE_PATH, FILE_TYPE, ROW_INDEX)
    codes = _parse_encoding(value)
    print("Validando códigos...")
    # Validaciones de rango
    assert codes.ndim == 1, "encoding no es vector 1D"
    assert codes.size > 0, "encoding vacío"
    mn, mx = int(codes.min()), int(codes.max())
    assert mn >= 0 and mx < 16384, f"Códigos fuera de rango [0,16383]: min={mn}, max={mx}"

    _print_stats(codes)

    print("Reshape de códigos a cuadrícula...")
    z_idx_1hw, (H, W), warn = _reshape_codes(codes)
    if warn:
        print(warn)
    print(f"Forma objetivo: (1,{H},{W})")

    # Guardar heatmap simple de índices para inspección rápida
    try:
        from PIL import Image
        grid = codes.reshape(H, W).astype(np.float32)
        gmin, gmax = float(grid.min()), float(grid.max())
        if gmax > gmin:
            norm = (grid - gmin) / (gmax - gmin)
        else:
            norm = np.zeros_like(grid)
        hm = (norm * 255).astype(np.uint8)
        Image.fromarray(hm, mode="L").resize((H*16, W*16), resample=Image.NEAREST).save(TOKENS_HEATMAP_PNG)
        print(f"Heatmap de tokens guardado en: {os.path.abspath(TOKENS_HEATMAP_PNG)}")
    except Exception as e:
        print(f"No pude guardar heatmap de tokens: {e}")

    if SKIP_DECODE:
        print("SKIP_DECODE=True -> se omite decodificación.")
    else:
        print("Decodificando con VQGAN unificado...")
        img = _decode_and_save(z_idx_1hw)
        # Estadísticas de reconstrucción
        try:
            vmin, vmax = float(img.min()), float(img.max())
            vmean, vstd = float(img.mean()), float(img.std())
            print(f"Recon stats [0,1]: min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f} std={vstd:.6f}")
        except Exception:
            pass
    print("Listo.")

if __name__ == "__main__":
    main()
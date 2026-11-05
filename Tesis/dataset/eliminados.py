import os
import hashlib
import sys
import shutil
from collections import defaultdict
from PIL import Image
import imagehash

# Ruta de la carpeta con tus imágenes
carpeta = r"C:\Users\Andrés Castro\Desktop\safe ai\train_data"

if not os.path.isdir(carpeta):
    print(f"❌ No se encontró la carpeta: {carpeta}")
    exit()

# Configuración
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PHASH_THRESHOLD = 3  # más bajo = más estricto
DRY_RUN = False       # True = no mueve, solo informa
QUARANTINE_DIR = os.path.join(carpeta, "_cuarentena")

def es_imagen(nombre):
    _, ext = os.path.splitext(nombre)
    return ext.lower() in ALLOWED_EXTS

def calcular_phash(path):
    with Image.open(path) as img:
        return imagehash.phash(img)

def bucket_key(h: imagehash.ImageHash, bits=16):
    return str(h)[:bits // 4]

archivos = [f for f in sorted(os.listdir(carpeta), key=str.lower) if os.path.isfile(os.path.join(carpeta, f)) and es_imagen(f)]

if not archivos:
    print("✅ No se encontraron imágenes válidas para analizar.")
    sys.exit(0)

representantes_por_bucket = defaultdict(list)  # bucket -> [(hash, nombre)]
duplicados_similares = []  # (dup, original, dist)

for nombre in archivos:
    ruta = os.path.join(carpeta, nombre)
    try:
        h = calcular_phash(ruta)
    except Exception as e:
        print(f"⚠️  No se pudo procesar {nombre}: {e}")
        continue

    bkey = bucket_key(h)
    candidatos = representantes_por_bucket.get(bkey, [])

    emparejado = False
    for h_ref, orig in candidatos:
        dist = (h - h_ref)
        if dist <= PHASH_THRESHOLD:
            duplicados_similares.append((nombre, orig, dist))
            emparejado = True
            break

    if not emparejado:
        representantes_por_bucket[bkey].append((h, nombre))

print(f"\n🔍 Posibles duplicados similares (pHash≤{PHASH_THRESHOLD}): {len(duplicados_similares)}")
if not duplicados_similares:
    print("✅ No se encontraron imágenes similares según el umbral actual.")
    sys.exit(0)

# Preparar cuarentena
if not DRY_RUN:
    os.makedirs(QUARANTINE_DIR, exist_ok=True)

moved = []
for dup, original, dist in duplicados_similares:
    src_img = os.path.join(carpeta, dup)
    if DRY_RUN:
        print(f"🛈 DRY-RUN: mover {dup} -> _cuarentena/ (similar a {original}, dist={dist})")
    else:
        try:
            dst_img = os.path.join(QUARANTINE_DIR, dup)
            shutil.move(src_img, dst_img)
            moved.append(dup)
            # Mover .txt asociado si existe
            base, _ = os.path.splitext(dup)
            txt_src = os.path.join(carpeta, base + ".txt")
            if os.path.isfile(txt_src):
                shutil.move(txt_src, os.path.join(QUARANTINE_DIR, base + ".txt"))
            print(f"📦 Movido a cuarentena: {dup} (similar a {original}, dist={dist})")
        except Exception as e:
            print(f"⚠️  Error moviendo {dup}: {e}")

if DRY_RUN:
    print("\nℹ️  DRY-RUN activo: no se movió ningún archivo. Cambia DRY_RUN=False para ejecutar.")
else:
    print(f"\n✅ Movimiento a cuarentena completo. Archivos movidos: {len(moved)}")
    if moved:
        for m in moved:
            print(f"  - {m}")

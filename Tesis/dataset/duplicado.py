import os
from collections import Counter

# Ruta a tu carpeta de imágenes
carpeta = r"C:\Users\Andrés Castro\Desktop\safe ai\train_data"

# Verifica si la ruta existe
if not os.path.isdir(carpeta):
    print(f"❌ No se encontró la carpeta: {carpeta}")
    exit()

# Lista los nombres de archivo en esa carpeta
nombres = [f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]
conteo_nombres = Counter(nombres)

# Filtra los duplicados
duplicados_nombre = {nombre: cuenta for nombre, cuenta in conteo_nombres.items() if cuenta > 1}

print(f"\n🔍 Archivos con nombres duplicados: {len(duplicados_nombre)}")
if duplicados_nombre:
    for nombre, cuenta in duplicados_nombre.items():
        print(f"  - {nombre}: {cuenta} veces")
else:
    print("✅ No hay archivos con nombres duplicados.")

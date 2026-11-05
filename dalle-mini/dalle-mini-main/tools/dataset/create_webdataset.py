import os
from pathlib import Path
import webdataset as wds
import logging

# Configuración de logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Configuración de Rutas y Parámetros
# El script asume que se ejecuta desde `dalle-mini-main/tools/dataset/`
# o que las rutas son relativas al directorio del proyecto si se ajusta.

# 1. Configuración de Rutas y Parámetros
# Rutas absolutas especificadas por el usuario para los datos y la salida.
source_data_dir = Path("C:\\Users\\Andrés Castro\\Desktop\\safe ai\\train_data")
output_shards_dir = Path("C:\\Users\\Andrés Castro\\Desktop\\safe ai\\output_shards")

# Crear el directorio de salida si no existe
output_shards_dir.mkdir(exist_ok=True)

files_per_shard = 1000  # Número de pares imagen-texto por archivo .tar
shard_name_pattern = "dataset_shard-%06d.tar"  # Patrón para nombrar los shards

logging.info(f"Directorio de datos fuente: {source_data_dir}")
logging.info(f"Directorio de salida para shards: {output_shards_dir}")
logging.info(f"Archivos por shard: {files_per_shard}")
logging.info(f"Patrón de nombre de shard: {shard_name_pattern}")

# 2. Listar y Emparejar Archivos
logging.info("Listando y emparejando archivos...")
image_files = sorted(list(source_data_dir.glob("*.jpg")))

if not image_files:
    logging.warning(f"No se encontraron archivos .jpg en {source_data_dir}. Verifica la ruta.")
    exit()

paired_files = []
for img_file_path in image_files:
    # El nombre base sin la extensión (ej. 'foto1')
    base_name = img_file_path.stem
    # Ruta al archivo de texto correspondiente
    txt_file_path = source_data_dir / f"{base_name}.txt"

    if txt_file_path.exists():
        paired_files.append((img_file_path, txt_file_path))
    else:
        logging.warning(f"No se encontró archivo .txt para {img_file_path.name} (se esperaba {txt_file_path.name}). Este archivo será omitido.")

if not paired_files:
    logging.error("No se encontraron pares de archivos imagen-texto válidos. Abortando.")
    exit()

logging.info(f"Se encontraron {len(paired_files)} pares de archivos imagen-texto.")

# 3. Crear los Shards
logging.info("Iniciando la creación de shards...")
shard_count = 0
file_count_in_current_shard = 0
current_shard_writer = None

try:
    for i, (img_path, txt_path) in enumerate(paired_files):
        if file_count_in_current_shard == 0:
            if current_shard_writer:
                current_shard_writer.close()
            
            current_shard_filename = shard_name_pattern % shard_count
            current_shard_full_path = output_shards_dir / current_shard_filename
            current_shard_writer = wds.TarWriter(str(current_shard_full_path))
            logging.info(f"Creando shard: {current_shard_full_path}")
            shard_count += 1

        key = img_path.stem  # Usar el nombre base del archivo como clave

        try:
            with open(img_path, "rb") as f_img:
                img_data = f_img.read()
            with open(txt_path, "r", encoding="utf-8") as f_txt:
                txt_data = f_txt.read()
        except Exception as e:
            logging.error(f"Error leyendo el par de archivos {img_path.name}/{txt_path.name}: {e}. Omitiendo este par.")
            continue

        sample = {
            "__key__": key,
            "jpg": img_data,
            "txt": txt_data,
        }
        
        try:
            current_shard_writer.write(sample)
        except Exception as e:
            logging.error(f"Error escribiendo el sample {key} al shard: {e}. Omitiendo este sample.")
            continue

        file_count_in_current_shard += 1
        if file_count_in_current_shard >= files_per_shard:
            file_count_in_current_shard = 0 # Reset para el próximo shard
            logging.info(f"Shard {current_shard_filename} completado.")

    if current_shard_writer:
        current_shard_writer.close()
        logging.info(f"Shard final {current_shard_filename} completado.")

    logging.info(f"Proceso completado. Se crearon {shard_count} shards en {output_shards_dir}")

except Exception as e:
    logging.error(f"Ocurrió un error inesperado durante la creación de shards: {e}")
finally:
    if current_shard_writer and not current_shard_writer.closed:
        current_shard_writer.close()


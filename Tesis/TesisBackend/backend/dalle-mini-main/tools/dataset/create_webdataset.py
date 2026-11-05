from pathlib import Path
import csv
import io
import webdataset as wds
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Configuración de Rutas y Parámetros
# Rutas absolutas para los datos y la salida.
source_data_dir = Path(r"C:\\Users\\Andrés Castro\\Desktop\\dataset-animals")  # raíz con subcarpetas e 'captions.csv'
output_shards_dir = Path(r"C:\\Users\\Andrés Castro\\Desktop\\Datasets de Prueba")

# Crear el directorio de salida si no existe
output_shards_dir.mkdir(exist_ok=True)

files_per_shard = 1000  # Número de pares imagen-texto por archivo .tar
shard_name_pattern = "dataset_shard-%06d.tar"  # Patrón para nombrar los shard
# Índice de caption a usar por imagen cuando hay múltiples (0..4). 0 => primera caption
desired_caption_index = 0

logging.info(f"Directorio de datos (raíz imágenes + captions.csv): {source_data_dir}")
logging.info(f"Directorio de salida para shards: {output_shards_dir}")
logging.info(f"Archivos por shard: {files_per_shard}")
logging.info(f"Patrón de nombre de shard: {shard_name_pattern}")

# 2. Listar y Emparejar Archivos
logging.info("Listando y emparejando archivos...")
# Aceptar .jpg, .jpeg y .png de forma recursiva (diagnóstico y fallback)
image_files = sorted(list(source_data_dir.rglob("*.jpg")) + list(source_data_dir.rglob("*.jpeg")) + list(source_data_dir.rglob("*.png")))

if not image_files:
    logging.warning(f"No se encontraron archivos .jpg/.jpeg/.png en {source_data_dir} (búsqueda recursiva). Verifica la ruta.")
    # No salimos; puede que el CSV apunte a rutas que existan pero no se detectaron por permisos/latencia

# Si existe un archivo captions.csv (header: image,caption[,raw]),
# creamos un sample por (imagen, caption). Si no, usamos el emparejamiento .jpg + .txt como antes.
captions_file = source_data_dir / "captions.csv"
use_csv_captions = captions_file.exists()

paired_items = []  # para modo CSV: (img_path, caption, cap_idx)
paired_files = []  # para modo TXT: (img_path, txt_path)

if use_csv_captions:
    logging.info(f"Detectado captions.csv en {captions_file}. Usando modo CSV (imagen x caption, subcarpetas soportadas)...")
    # Índices de imágenes
    # - por nombre base (minúsculas) para fallback por nombre
    images_by_name = {p.name.lower(): p for p in image_files}
    # - por ruta relativa desde source_data_dir en formato posix y en minúsculas (p.ej. 'abra/abra_0001.jpg')
    def _rel_posix_lower(p: Path) -> str:
        try:
            return p.relative_to(source_data_dir).as_posix().lower()
        except Exception:
            return p.name.lower()
    images_by_rel = {_rel_posix_lower(p): p for p in image_files}
    # Leer captions
    counters = {"missing_images": 0, "total_rows": 0}
    unmatched_preview = []  # ejemplos de filas sin imagen para diagnóstico
    # Conteo de ocurrencias por imagen y set de seleccionadas
    selected_images = set()
    try:
        # Abrir CSV con detección de encoding (utf-8-sig, utf-8, cp1252, latin-1)
        # y detección de delimitador (coma o punto y coma).
        def _open_csv_with_fallback(path: Path):
            for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                try:
                    f = open(path, "r", encoding=enc, newline="")
                    text = f.read()
                    # Normalizar comillas curvas a comillas rectas para que el parser CSV las reconozca
                    # “ ” -> "   ‘ ’ -> '
                    text = (
                        text.replace("\u201c", '"')
                            .replace("\u201d", '"')
                            .replace("\u2018", "'")
                            .replace("\u2019", "'")
                    )
                    sample = text[:4096]
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                    except Exception:
                        dialect = csv.excel
                    reader_local = csv.reader(io.StringIO(text), dialect)
                    # Retornamos también el texto normalizado para pruebas con delimitador alterno
                    return f, reader_local, enc, getattr(dialect, 'delimiter', ','), text
                except UnicodeDecodeError:
                    try:
                        f.close()
                    except Exception:
                        pass
                    continue
            raise UnicodeDecodeError("csv", b"", 0, 1, "No se pudo decodificar el CSV con encodings comunes")

        f, reader, enc_used, delim_used, normalized_text = _open_csv_with_fallback(captions_file)
        try:
            # Leer con el delimitador detectado
            rows_a = list(reader)
            good_a = sum(1 for r in rows_a if len(r) >= 2)
            logging.info(f"CSV encoding detectado: {enc_used}, delimitador detectado: '{delim_used}', filas: {len(rows_a)}, válidas: {good_a}")
            # Probar el delimitador alterno y elegir el mejor
            alt_delim = ';' if delim_used == ',' else ','
            reader_alt = csv.reader(io.StringIO(normalized_text), delimiter=alt_delim)
            rows_b = list(reader_alt)
            good_b = sum(1 for r in rows_b if len(r) >= 2)
            logging.info(f"Prueba delimitador alterno '{alt_delim}': filas: {len(rows_b)}, válidas: {good_b}")

            if good_b > good_a:
                rows = rows_b
                chosen_delim = alt_delim
                good_rows = good_b
            else:
                rows = rows_a
                chosen_delim = delim_used
                good_rows = good_a
            logging.info(f"Delimitador elegido: '{chosen_delim}', filas válidas: {good_rows}/{len(rows)}")

            # Fallback: si aún hay pocas filas válidas, parsear manualmente por la primera coma
            threshold = max(500, int(0.8 * max(1, len(rows))))
            if good_rows < threshold:
                logging.warning("Pocas filas válidas con parser CSV; usando parser manual por primera coma como fallback.")
                lines = normalized_text.splitlines()
                # detect header
                start = 1 if (lines and lines[0].lower().startswith('image')) else 0
                manual_rows = []
                for ln in lines[start:]:
                    if not ln.strip():
                        continue
                    idx = ln.find(',')
                    if idx == -1:
                        continue
                    img = ln[:idx].strip()
                    cap = ln[idx+1:].strip()
                    # quitar comillas exteriores si existen
                    if len(cap) >= 2 and cap[0] == '"' and cap[-1] == '"':
                        cap = cap[1:-1]
                    manual_rows.append([img, cap])
                logging.info(f"Parser manual por primera coma: filas: {len(manual_rows)}")
                rows = manual_rows

            first_row = rows[0] if rows else None
            # Detectar si hay header
            has_header = (
                first_row is not None
                and len(first_row) >= 2
                and first_row[0].strip().lower() == "image"
            )
            # Si no hay header, procesar también la primera fila como dato
            def process_row(row, cap_index_map):
                if not row or len(row) < 2:
                    return
                # Normalizar ruta relativa del CSV; soportar posibles backslashes
                img_name_raw = row[0].strip()
                img_rel_norm = img_name_raw.replace('\\', '/').replace('\\', '/').strip('/').lower()
                img_name = Path(img_name_raw).name.lower()
                # columna 1 = caption; si hay columna 'raw' ignoramos
                caption = row[1].strip()
                # Limpieza de comillas en caption: curvas -> rectas, colapsar "" -> ", y quitar comillas exteriores
                caption = (
                    caption.replace("\u201c", '"')
                           .replace("\u201d", '"')
                           .replace("\u2018", "'")
                           .replace("\u2019", "'")
                )
                caption = caption.replace('""', '"')
                # quitar comillas exteriores repetidas
                while len(caption) >= 2 and caption[0] == '"' and caption[-1] == '"':
                    caption = caption[1:-1].strip()
                if not img_name or not caption:
                    return
                counters["total_rows"] += 1
                # 1) intento directo por ruta relativa desde source_data_dir
                img_path = (source_data_dir / img_name_raw).resolve()
                if not img_path.exists():
                    # 2) intento por ruta relativa normalizada (posix lower)
                    img_path = images_by_rel.get(img_rel_norm)
                if img_path is None or not Path(img_path).exists():
                    # 3) intento con otras extensiones comunes por nombre base
                    img_path = None
                    stem = Path(img_name).stem.lower()
                    for ext_try in (".png", ".jpg", ".jpeg"):
                        alt_name = f"{stem}{ext_try}"
                        img_path = images_by_name.get(alt_name)
                        if img_path is not None:
                            break
                if img_path is None:
                    counters["missing_images"] += 1
                    if len(unmatched_preview) < 10:
                        unmatched_preview.append(img_name_raw)
                    return
                # Contador 0-based de captions vistas para esta imagen
                current_idx = cap_index_map.get(img_name, 0)
                # Si coincide con el índice deseado y aún no seleccionada, la usamos
                if current_idx == desired_caption_index and img_name not in selected_images:
                    cap_idx = desired_caption_index
                    paired_items.append((img_path, caption, cap_idx))
                    selected_images.add(img_name)
                # Incrementar contador para esta imagen
                cap_index_map[img_name] = current_idx + 1

            cap_index_map = {}
            start_idx = 1 if has_header else 0
            for row in rows[start_idx:]:
                process_row(row, cap_index_map)
        finally:
            f.close()
    except Exception as e:
        logging.error(f"Error leyendo captions.csv: {e}")
        exit()

    if counters["missing_images"]:
        logging.warning(
            f"{counters['missing_images']} filas en captions.csv no tienen imagen correspondiente bajo {source_data_dir}."
        )
        if unmatched_preview:
            logging.warning(f"Ejemplos de nombres no encontrados (hasta 10): {unmatched_preview}")

    if not paired_items:
        logging.error("No se pudieron emparejar imágenes con captions. Abortando.")
        exit()

    logging.info(
        f"Total de filas en captions: {counters['total_rows']}. Total de samples (imagen x caption): {len(paired_items)}"
    )
else:
    logging.info("No se encontró captions.csv. Usando modo clásico (.jpg + .txt) en subcarpetas no soportado por defecto.")
    for img_file_path in image_files:
        # El nombre base sin la extensión (ej. 'foto1')
        base_name = img_file_path.stem
        # Ruta al archivo de texto correspondiente
        txt_file_path = source_data_dir / f"{base_name}.txt"

        if txt_file_path.exists():
            paired_files.append((img_file_path, txt_file_path))
        else:
            logging.warning(
                f"No se encontró archivo .txt para {img_file_path.name} (se esperaba {txt_file_path.name}). Este archivo será omitido."
            )

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
    if use_csv_captions:
        for i, (img_path, caption, cap_idx) in enumerate(paired_items):
            if file_count_in_current_shard == 0:
                if current_shard_writer:
                    current_shard_writer.close()
                
                current_shard_filename = shard_name_pattern % shard_count
                current_shard_full_path = output_shards_dir / current_shard_filename
                current_shard_writer = wds.TarWriter(open(current_shard_full_path, "wb"))
                logging.info(f"Creando shard: {current_shard_full_path}")
                shard_count += 1

            # Usar solo el nombre base de la imagen como clave (sin sufijo de índice)
            key = img_path.stem

            try:
                with open(img_path, "rb") as f_img:
                    img_data = f_img.read()
            except Exception as e:
                logging.error(f"Error leyendo la imagen {img_path.name}: {e}. Omitiendo este sample.")
                continue

            # Usar la extensión real del archivo (png/jpg/jpeg) como clave de la imagen
            ext = img_path.suffix.lower().lstrip('.') or 'jpg'
            sample = {
                "__key__": key,
                ext: img_data,
                "txt": caption,
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
    else:
        for i, (img_path, txt_path) in enumerate(paired_files):
            if file_count_in_current_shard == 0:
                if current_shard_writer:
                    current_shard_writer.close()
                
                current_shard_filename = shard_name_pattern % shard_count
                current_shard_full_path = output_shards_dir / current_shard_filename
                current_shard_writer = wds.TarWriter(open(current_shard_full_path, "wb"))
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

            ext = img_path.suffix.lower().lstrip('.') or 'jpg'
            sample = {
                "__key__": key,
                ext: img_data,
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
    if current_shard_writer:
        current_shard_writer.close()


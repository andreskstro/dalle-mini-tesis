#!/usr/bin/env python3
"""
Offline top-k per neuron for DALL·E mini (hidden before lm_head)
- Prefilter (forward-only): keep top-N candidates per neuron by activation h_j(x)
- Refinement (VJP): compute influence s_j(x) = ||\u2202F_j/\u2202\u03b8_j||_2 with F_j(h)=h_j for candidates, keep top-5

Inputs (dataset): Parquet files with columns: filename (str), caption (str), encoding (list[int] length 256)

Usage (sin parámetros):
  python tools/offline/compute_topk_per_neuron.py

También puedes sobre-escribir con flags si lo deseas:
  python tools/offline/compute_topk_per_neuron.py \
    --dataset /home/kstro/proyectos/dalle-mini/dalle-mini-main/tools/dataset/encoded_data/train_normal \
    --out tools/offline/offline_topk_lm_head.jsonl \
    --k 5 --prefilter-n 200 --top-m-per-image 256 --batch-size 128 --chunk-size 256 \
    --model-path /home/kstro/proyectos/dalle-mini/dalle-mini-main/outputs/dalle-mini-finetuned \
    --processor-id dalle-mini/dalle-mini --use-snapshot 0

Notes:
- Requires: jax, dalle_mini, vqgan_jax, pyarrow (for parquet), numpy
- Uses loader.get_bundle() to load model/params and precomputed W_lm & pinv_W if available
"""
from __future__ import annotations
import os
import sys
import json
import math
import argparse
import logging
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Any

import numpy as np

# Optional imports (dataset reader)
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# Defaults embebidos (modifica aquí si deseas cambiar la configuración por defecto)
DEFAULT_DATASET = \
    "/home/kstro/proyectos/dalle-mini/dalle-mini-main/tools/dataset/encoded_data/train_normal"
DEFAULT_OUT = "tools/analyze/offline_topk_lm_head.jsonl"
DEFAULT_K = 5
DEFAULT_PREFILTER_N = 80
DEFAULT_TOP_M_PER_IMAGE = 256
DEFAULT_BATCH_SIZE = 128
DEFAULT_CHUNK_SIZE = 256
DEFAULT_MODEL_PATH = \
    "/home/kstro/proyectos/dalle-mini/dalle-mini-main/outputs/dalle-mini-256-2"
DEFAULT_PROCESSOR_ID = "dalle-mini/dalle-mini"
DEFAULT_USE_SNAPSHOT = 0
DEFAULT_VJP_BATCH = 64
DEFAULT_MODE = "full"  # full | prefilter_only | vjp_only
DEFAULT_PREFILTER_OUT = "tools/analyze/offline_candidates_lm_head.jsonl"
DEFAULT_ENC_LEN = 64  # longitud fija para encoder (caption), para shapes estables en JIT

# Project imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import loader  # noqa: E402
import models_config as cfg  # noqa: E402

logger = logging.getLogger("offline_topk")
logger.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)


def iter_parquet_rows(dataset_dir: str, *, quiet: bool = False):
    if pq is None:
        raise RuntimeError("pyarrow no instalado. Instala 'pyarrow' para leer Parquet")
    # Acepta un archivo parquet único o un directorio con múltiples parquets
    files: List[str] = []
    if os.path.isdir(dataset_dir):
        for name in sorted(os.listdir(dataset_dir)):
            if name.endswith(".parquet"):
                files.append(os.path.join(dataset_dir, name))
    else:
        if dataset_dir.endswith(".parquet"):
            files = [dataset_dir]
        else:
            raise RuntimeError(f"Ruta de dataset inválida: {dataset_dir}")

    for f in files:
        try:
            if not quiet:
                logger.info("Leyendo parquet: %s", f)
            tbl = pq.read_table(f, columns=["filename", "caption", "encoding"])  # type: ignore
            cols = {c: i for i, c in enumerate(tbl.column_names)}
            fn_col = cols.get("filename")
            cap_col = cols.get("caption")
            enc_col = cols.get("encoding")
            if fn_col is None or enc_col is None:
                raise RuntimeError("Parquet debe contener columnas 'filename' y 'encoding'")
            fn_arr = tbl.column(fn_col)
            cap_arr = tbl.column(cap_col) if cap_col is not None else None
            enc_arr = tbl.column(enc_col)
            for i in range(tbl.num_rows):
                filename = fn_arr[i].as_py()
                caption = cap_arr[i].as_py() if cap_arr is not None else None
                encoding = enc_arr[i].as_py()
                yield {"filename": filename, "caption": caption, "encoding": encoding}
        except Exception as e:
            logger.warning("salteando parquet %s por error: %s", f, e)


def count_parquet_rows(dataset_dir: str) -> int:
    """Cuenta filas totales en los parquets para configurar la barra de progreso."""
    if pq is None:
        return 0
    total = 0
    files: List[str] = []
    if os.path.isdir(dataset_dir):
        for name in os.listdir(dataset_dir):
            if name.endswith(".parquet"):
                files.append(os.path.join(dataset_dir, name))
    else:
        if dataset_dir.endswith(".parquet"):
            files = [dataset_dir]
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            total += pf.metadata.num_rows
        except Exception:
            pass
    return total


def topk_push(heap: List[Tuple[float, Any]], k: int, item: Tuple[float, Any]):
    import heapq
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def run(dataset_dir: str, out_path: str, k: int, prefilter_n: int, top_m_per_image: int,
        batch_size: int, chunk_size: int):
    # Load model bundle
    bundle = loader.get_bundle(force_reload=False)
    model = bundle.get("model")
    params = bundle.get("params")
    processor = bundle.get("processor")
    W_lm = bundle.get("W_lm")
    pinv_W = bundle.get("pinv_W")
    if model is None or params is None or processor is None or W_lm is None or pinv_W is None:
        raise RuntimeError("Bundle incompleto. Revisa loader: model/params/processor/W_lm/pinv_W")

    import jax
    import jax.numpy as jnp
    from jax import nn as jnn

    # Prepare caches
    # For each neuron index (int), store a min-heap of size `prefilter_n` with (activation, filename, payload)
    candidate_heaps: Dict[int, List[Tuple[float, str, Dict[str, Any]]]] = defaultdict(list)
    # Cache per filename -> prebuilt input dict
    sample_cache: Dict[str, Dict[str, Any]] = {}

    # Helper: build inputs for a single sample (caption + encoding)
    def build_inputs(sample: Dict[str, Any]) -> Dict[str, Any]:
        caption = sample.get("caption") or ""
        tokenized = processor([caption])
        # padding/truncado manual a longitud fija para estabilidad del JIT
        pad_id = None
        try:
            pad_id = int(getattr(getattr(processor, 'tokenizer', processor), 'pad_token_id', 0))
        except Exception:
            pad_id = 0
        enc_len = int(DEFAULT_ENC_LEN)
        ids_1d = np.asarray(tokenized["input_ids"][0], dtype=np.int32)
        mask_1d = np.asarray(tokenized.get("attention_mask", [[1]*len(ids_1d)])[0], dtype=np.int32)
        # truncar
        ids_1d = ids_1d[:enc_len]
        mask_1d = mask_1d[:enc_len]
        # pad hasta enc_len
        if ids_1d.shape[0] < enc_len:
            pad_needed = enc_len - ids_1d.shape[0]
            ids_1d = np.concatenate([ids_1d, np.full((pad_needed,), pad_id, dtype=np.int32)], axis=0)
            mask_1d = np.concatenate([mask_1d, np.zeros((pad_needed,), dtype=np.int32)], axis=0)
        enc_input_ids = jnp.array(ids_1d[None, :])
        enc_attention_mask = jnp.array(mask_1d[None, :])
        pos_enc = jnp.arange(enc_len, dtype=jnp.int32)[None, :]
        # encoding: list[256] -> jnp array [1, 256]
        enc_tokens = np.asarray(sample["encoding"], dtype=np.int32)
        if enc_tokens.size != 256:
            raise RuntimeError(f"encoding inválido, esperado 256 tokens, got {enc_tokens.size}")
        gen_tokens_no_bos = jnp.array(enc_tokens[None, :], dtype=jnp.int32)
        start_id = int(getattr(getattr(model, "config", object()), "decoder_start_token_id", 0))
        decoder_input_ids = jnp.concatenate(
            [jnp.array([[start_id]], dtype=jnp.int32), gen_tokens_no_bos[:, :-1]], axis=1
        )
        decoder_attention_mask = jnp.ones_like(decoder_input_ids, dtype=jnp.int32)
        pos_dec = jnp.arange(decoder_input_ids.shape[1], dtype=jnp.int32)[None, :]
        return {
            "input_ids": enc_input_ids,
            "attention_mask": enc_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "position_ids": pos_enc,
            "decoder_position_ids": pos_dec,
            "gen_tokens_no_bos": gen_tokens_no_bos,
            "_filename": sample.get("filename"),
            "_caption": sample.get("caption"),
        }

    # Forward -> obtener hidden nativo previo a lm_head para un batch de samples (list size B)
    def forward_h_last(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        # stack inputs
        enc_input_ids = jnp.concatenate([s["input_ids"] for s in samples], axis=0)
        enc_attention_mask = jnp.concatenate([s["attention_mask"] for s in samples], axis=0)
        decoder_input_ids = jnp.concatenate([s["decoder_input_ids"] for s in samples], axis=0)
        decoder_attention_mask = jnp.concatenate([s["decoder_attention_mask"] for s in samples], axis=0)
        position_ids = jnp.concatenate([s["position_ids"] for s in samples], axis=0)
        decoder_position_ids = jnp.concatenate([s["decoder_position_ids"] for s in samples], axis=0)

        inputs = {
            "input_ids": enc_input_ids,
            "attention_mask": enc_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "position_ids": position_ids,
            "decoder_position_ids": decoder_position_ids,
        }
        # Capturar intermedios para extraer el hidden nativo [B, seq, H]
        y_out_top, inter = model.module.apply({"params": params}, **inputs, capture_intermediates=True, mutable=["intermediates"])
        logits_last = jnp.array(y_out_top.logits[:, -1, :])  # [B, vocab]

        # Heurística: tomar el último tensor con forma [B, seq, H]
        def pick_hidden_native(intermediates):
            candidates = []
            try:
                from collections.abc import Mapping
                def walk(d):
                    if isinstance(d, (list, tuple)):
                        for x in d:
                            walk(x)
                    elif isinstance(d, Mapping):
                        for _, v in d.items():
                            walk(v)
                    else:
                        try:
                            arr = jnp.asarray(d)
                            if arr.ndim == 3 and arr.shape[0] == enc_input_ids.shape[0]:
                                candidates.append(arr)
                        except Exception:
                            pass
                walk(inter)
            except Exception:
                candidates = []
            if not candidates:
                return None
            h3 = candidates[-1]
            try:
                return h3[:, -1, :]  # [B, H]
            except Exception:
                return None

        h_last_native = pick_hidden_native(inter)
        if h_last_native is not None:
            h_last = h_last_native
        else:
            # Fallback: reconstrucción
            h_last = logits_last @ pinv_W
        return np.asarray(h_last), np.asarray(logits_last)

    # Prefilter pass ---------------------------------------------------------
    mode = getattr(args, 'mode', 'full') if 'args' in globals() else 'full'
    start_t = time.perf_counter()
    if mode != 'vjp_only':
        logger.info("Prefiltro (solo forward): dataset=%s", dataset_dir)
        batch_buf: List[Dict[str, Any]] = []
        seen = 0
        total_rows = count_parquet_rows(dataset_dir)
        pbar = None
        if tqdm is not None and total_rows > 0:
            pbar = tqdm(total=total_rows, desc="Prefiltro samples", unit="img")
        def process_batch(buf: List[Dict[str, Any]]):
            if not buf:
                return
            h_last, _ = forward_h_last(buf)  # [B, hidden]
            B, H = h_last.shape
            M = min(top_m_per_image, H)
            # For each image in batch, pick top-M neuron indices by activation
            for bi in range(B):
                row = h_last[bi]
                top_idx = np.argpartition(row, -M)[-M:]
                # refine order
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
                fn = buf[bi].get("_filename")
                payload = {
                    "filename": fn,
                    "caption": buf[bi].get("_caption"),
                }
                for j in top_idx:
                    act = float(row[j])
                    # add filename as tiebreaker to avoid comparing dicts on equal scores
                    topk_push(candidate_heaps[j], prefilter_n, (act, str(fn) if fn is not None else "", payload))
            # progress
            if pbar is not None:
                pbar.update(B)
                elapsed = time.perf_counter() - start_t
                pbar.set_postfix_str(f"elapsed {elapsed/60:.1f}m")

        try:
            for rec in iter_parquet_rows(dataset_dir):
                filename = rec["filename"]
                try:
                    if filename not in sample_cache:
                        sample_cache[filename] = build_inputs(rec)
                    batch_buf.append(sample_cache[filename])
                    seen += 1
                    if len(batch_buf) >= batch_size:
                        process_batch(batch_buf)
                        batch_buf = []
                        if seen % (batch_size * 10) == 0 and pbar is None:
                            elapsed = time.perf_counter() - start_t
                            logger.info("Prefiltro procesadas=%d | elapsed=%.1f min", seen, elapsed/60.0)
                except Exception as e:
                    logger.warning("salteando %s por error: %s", filename, e)
                    continue
            # flush remainder
            process_batch(batch_buf)
        except KeyboardInterrupt:
            logger.warning("Interrumpido por usuario, continuando con lo acumulado…")
        finally:
            if pbar is not None:
                pbar.close()

    # Guardar candidatos del prefiltro (por neurona)
    # Estructura: una línea por neurona con {neuron, candidates:[{filename, caption, act}]}
    def write_prefilter_jsonl(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as pf:
            for neuron in sorted(candidate_heaps.keys()):
                heap = candidate_heaps[neuron]
                heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)[:prefilter_n]
                cands = []
                for act, _fn, payload in heap_sorted:
                    cands.append({
                        "filename": payload.get("filename"),
                        "caption": payload.get("caption"),
                        "act": float(act),
                    })
                pf.write(json.dumps({"neuron": int(neuron), "candidates": cands}, ensure_ascii=False) + "\n")

    # ------------------------
    # Refinamiento VJP (influencia fiel)
    # ------------------------
    # Si el modo es prefilter_only, escribir candidatos y salir
    if 'args' in globals() and getattr(args, 'mode', 'full') == 'prefilter_only':
        logger.info("Escribiendo candidatos del prefiltro en: %s", args.prefilter_out)
        write_prefilter_jsonl(args.prefilter_out)
        elapsed_total = time.perf_counter() - start_t
        logger.info("Prefiltro completado | tiempo total: %.1f min", elapsed_total/60.0)
        return

    logger.info("Refinamiento VJP: calculando influencia fiel por neurona y candidato…")
    # Helpers para gradiente y norma L2 excluyendo lm_head
    import jax
    import jax.numpy as jnp

    # Construir un "máscara" de parámetros que excluya lm_head, para usar dentro de JIT
    def _build_exclusion_mask(tree):
        if isinstance(tree, dict):
            out = {}
            for k, v in tree.items():
                if k == "lm_head":
                    out[k] = jax.tree_map(lambda x: jnp.zeros_like(x), v)
                else:
                    out[k] = _build_exclusion_mask(v)
            return out
        else:
            return jnp.ones_like(jnp.asarray(tree))

    exclusion_mask = _build_exclusion_mask(params)

    # forward single that returns h_last vector
    def forward_h_last_single(p, sample_inputs) -> jnp.ndarray:
        inputs = {
            "input_ids": sample_inputs["input_ids"],
            "attention_mask": sample_inputs["attention_mask"],
            "decoder_input_ids": sample_inputs["decoder_input_ids"],
            "decoder_attention_mask": sample_inputs["decoder_attention_mask"],
            "position_ids": sample_inputs["position_ids"],
            "decoder_position_ids": sample_inputs["decoder_position_ids"],
        }
        y_out_top, inter = model.module.apply({"params": p}, **inputs, capture_intermediates=True, mutable=["intermediates"])
        logits_last = jnp.array(y_out_top.logits[0, -1, :])  # [vocab]

        # Intentar extraer hidden nativo [1, seq, H]
        def pick_hidden_native_single(intermediates):
            candidates = []
            try:
                from collections.abc import Mapping
                def walk(d):
                    if isinstance(d, (list, tuple)):
                        for x in d:
                            walk(x)
                    elif isinstance(d, Mapping):
                        for _, v in d.items():
                            walk(v)
                    else:
                        try:
                            arr = jnp.asarray(d)
                            if arr.ndim == 3 and arr.shape[0] == 1:
                                candidates.append(arr)
                        except Exception:
                            pass
                walk(inter)
            except Exception:
                candidates = []
            if not candidates:
                return None
            h3 = candidates[-1]
            try:
                return h3[0, -1, :]  # [H]
            except Exception:
                return None

        h_native = pick_hidden_native_single(inter)
        if h_native is not None:
            return h_native
        # Fallback reconstrucción
        return logits_last @ jnp.asarray(pinv_W)

    # Single compiled batched grad-norm over samples; neuron is passed como one-hot para evitar recompiles
    hidden_dim = int(W_lm.shape[0])

    def make_batched_grad_norm():
        # Calcula gradiente e inmediatamente su norma L2 excluyendo lm_head, todo dentro de JIT
        def norm_fn_one(pp, sample_inputs, e_vec):
            def scalar_fn(p_inner):
                h = forward_h_last_single(p_inner, sample_inputs)  # [hidden]
                return jnp.vdot(h, e_vec)  # escalar = <h, e_j>

            grads = jax.grad(scalar_fn)(pp)

            # Aplicar máscara de exclusión (0 en lm_head, 1 en el resto) y sumar cuadrados
            def leaf_sumsq(g, m):
                g = jnp.asarray(g)
                m = jnp.asarray(m)
                return jnp.sum((g * m) ** 2)

            per_leaf = jax.tree_util.tree_leaves(jax.tree_map(leaf_sumsq, grads, exclusion_mask))
            total = jnp.add_n(per_leaf) if per_leaf else jnp.array(0.0, dtype=jnp.float32)
            return jnp.sqrt(total)

        return jax.jit(jax.vmap(norm_fn_one, in_axes=(None, 0, None)))

    _grad_norm_batched = make_batched_grad_norm()

    def influence_scores_for_batch(p, batch_inputs, e_vec: jnp.ndarray):
        # Devuelve directamente vector [B] de normas L2 excluyendo lm_head
        return _grad_norm_batched(p, batch_inputs, e_vec)

    # Prepara lista de trabajo según el modo
    work: List[Tuple[int, List[Dict[str, Any]]]] = []
    total_candidates = 0
    mode = getattr(args, 'mode', 'full') if 'args' in globals() else 'full'
    if mode == 'vjp_only':
        # leer candidatos desde archivo args.prefilter_out
        prefilter_path = getattr(args, 'prefilter_out', DEFAULT_PREFILTER_OUT)
        logger.info("Leyendo candidatos desde: %s", prefilter_path)
        needed: set[str] = set()
        with open(prefilter_path, "r", encoding="utf-8") as pf:
            for line in pf:
                rec = json.loads(line)
                neuron = int(rec.get("neuron"))
                cands = rec.get("candidates", [])
                cand_list = []
                for it in cands[:prefilter_n]:
                    fn = it.get("filename")
                    cap = it.get("caption")
                    cand_list.append({"filename": fn, "caption": cap})
                    if fn:
                        needed.add(fn)
                if cand_list:
                    work.append((neuron, cand_list))
                    total_candidates += len(cand_list)
        # construir sample_cache SOLO para los filenames necesarios escaneando el dataset una vez
        if needed:
            logger.info("Construyendo cache para %d candidatos necesarios…", len(needed))
            for rec in iter_parquet_rows(dataset_dir, quiet=True):
                fn = rec["filename"]
                if fn in needed and fn not in sample_cache:
                    sample_cache[fn] = build_inputs(rec)
                    needed.remove(fn)
                    if tqdm is None and len(sample_cache) % 500 == 0:
                        logger.info("cache construida: %d", len(sample_cache))
                if not needed:
                    break
            if needed:
                logger.warning("No se encontraron %d filenames en el dataset (se omitirán en VJP)", len(needed))
    else:
        # modo full: usar candidate_heaps en memoria y (opcional) escribir archivo de prefiltro
        if 'args' in globals() and getattr(args, 'prefilter_out', None):
            try:
                logger.info("Escribiendo candidatos del prefiltro en: %s", args.prefilter_out)
                write_prefilter_jsonl(args.prefilter_out)
            except Exception as e:
                logger.warning("No se pudo escribir prefilter_out: %s", e)
        for neuron in sorted(candidate_heaps.keys()):
            heap = candidate_heaps[neuron]
            heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)[:prefilter_n]
            cand_list = []
            for _score, _fn, payload in heap_sorted:
                cand_list.append({
                    "filename": payload.get("filename"),
                    "caption": payload.get("caption"),
                })
            if cand_list:
                work.append((int(neuron), cand_list))
                total_candidates += len(cand_list)

    # Progreso VJP
    vjp_pbar = None
    if tqdm is not None and total_candidates > 0:
        vjp_pbar = tqdm(total=total_candidates, desc="VJP refine", unit="cand")
    vjp_start = time.perf_counter()
    # JIT warmup: compilar una vez el backward con un batch mínimo
    try:
        # Busca una muestra válida
        warm_sample = None
        for _, cand_list in work:
            for item in cand_list:
                si = sample_cache.get(item.get("filename"))
                if si is not None:
                    warm_sample = si
                    break
            if warm_sample is not None:
                break
        if warm_sample is not None:
            warm_batch = {
                k: jnp.concatenate([warm_sample[k]], axis=0) for k in [
                    "input_ids", "attention_mask", "decoder_input_ids",
                    "decoder_attention_mask", "position_ids", "decoder_position_ids",
                ]
            }
            e0 = jnn.one_hot(0, hidden_dim, dtype=jnp.float32)
            _ = influence_scores_for_batch(params, warm_batch, e0)
    except Exception:
        pass

    # Calcula scores fieles y escribe salida final top-k por neurona
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for neuron, cand_list in work:
            refined_heap: List[Tuple[float, str, Dict[str, Any]]] = []  # (score, filename, payload)
            j_idx = int(neuron)
            # Procesar en batches de candidatos para compilar una sola vez
            Bv = DEFAULT_VJP_BATCH
            # Construir batches apilando los dicts de sample_inputs
            def stack_batch(samples_slice: List[Dict[str, Any]]):
                fields = [
                    "input_ids", "attention_mask", "decoder_input_ids",
                    "decoder_attention_mask", "position_ids", "decoder_position_ids",
                ]
                stacked = {}
                for key in fields:
                    stacked[key] = jnp.concatenate([s[key] for s in samples_slice], axis=0)
                return stacked

            # generar lista de sample_inputs para todos los candidatos válidos
            valid_samples: List[Dict[str, Any]] = []
            valid_payloads: List[Dict[str, Any]] = []
            for item in cand_list:
                fn = item.get("filename")
                si = sample_cache.get(fn)
                if si is None:
                    continue
                valid_samples.append(si)
                valid_payloads.append(item)

            for start in range(0, len(valid_samples), Bv):
                chunk_samples = valid_samples[start:start+Bv]
                if not chunk_samples:
                    continue
                n_valid = len(chunk_samples)
                # Pad chunk to fixed Bv by repeating last sample to keep shapes constant
                if n_valid < Bv:
                    last = chunk_samples[-1]
                    chunk_samples = chunk_samples + [last] * (Bv - n_valid)
                batch_inputs = stack_batch(chunk_samples)
                try:
                    e_vec = jnn.one_hot(j_idx, hidden_dim, dtype=jnp.float32)
                    scores_vec = influence_scores_for_batch(params, batch_inputs, e_vec)
                    scores_list = list(np.asarray(scores_vec)[:n_valid].tolist())
                    for score, payload in zip(scores_list, valid_payloads[start:start+n_valid]):
                        fn2 = payload.get("filename")
                        topk_push(refined_heap, k, (float(score), str(fn2) if fn2 else "", payload))
                        if vjp_pbar is not None:
                            vjp_pbar.update(1)
                except Exception as e:
                    logger.warning("VJP batch fallo neurona=%d: %s", j_idx, e)
                    # fallback individual si el batch falla
                    for payload in valid_payloads[start:start+n_valid]:
                        fn2 = payload.get("filename")
                        si2 = sample_cache.get(fn2)
                        if si2 is None:
                            continue
                        try:
                            # Fallback: grad-norm individual, reutilizando la misma máscara de exclusión
                            def scalar_fn(pp):
                                h = forward_h_last_single(pp, si2)
                                return h[j_idx]

                            grads_single = jax.grad(scalar_fn)(params)

                            def leaf_sumsq(g, m):
                                g = jnp.asarray(g)
                                m = jnp.asarray(m)
                                return jnp.sum((g * m) ** 2)

                            per_leaf = jax.tree_util.tree_leaves(
                                jax.tree_map(leaf_sumsq, grads_single, exclusion_mask)
                            )
                            total = float(np.sqrt(float(np.add.reduce(np.array(per_leaf))))) if per_leaf else 0.0
                            topk_push(refined_heap, k, (total, str(fn2) if fn2 else "", payload))
                        except Exception:
                            continue
                        if vjp_pbar is not None:
                            vjp_pbar.update(1)
            # ordenar final descendentemente y escribir línea
            refined_sorted = sorted(refined_heap, key=lambda x: x[0], reverse=True)[:k]
            out_items = []
            for score, _fn, payload in refined_sorted:
                out_items.append({
                    "filename": payload.get("filename"),
                    "score": float(score),
                    "caption": payload.get("caption"),
                })
            f.write(json.dumps({"neuron": neuron, "topk": out_items}, ensure_ascii=False) + "\n")

    if vjp_pbar is not None:
        vjp_pbar.close()
    elapsed_total = time.perf_counter() - start_t
    elapsed_vjp = time.perf_counter() - vjp_start
    processed_neurons = len(work) if len(work) > 0 else len(candidate_heaps)
    logger.info("Listo. Neuronas procesadas: %d | tiempo total: %.1f min (VJP: %.1f min)",
                processed_neurons, elapsed_total/60.0, elapsed_vjp/60.0)
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, default=DEFAULT_DATASET,
                    help="Ruta parquet o carpeta con parquets")
    ap.add_argument("--out", required=False, default=DEFAULT_OUT,
                    help="Archivo JSONL de salida")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--prefilter-n", type=int, default=DEFAULT_PREFILTER_N)
    ap.add_argument("--top-m-per-image", type=int, default=DEFAULT_TOP_M_PER_IMAGE)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                    help="Ruta local del modelo DALL·E mini (sobre-escribe models_config.DALLE_MODEL)")
    ap.add_argument("--processor-id", type=str, default=DEFAULT_PROCESSOR_ID,
                    help="Repo/ID del processor en HF Hub (sobre-escribe models_config.DALLE_PROCESSOR)")
    ap.add_argument("--use-snapshot", type=int, default=DEFAULT_USE_SNAPSHOT,
                    help="1 para snapshot_download desde HF Hub, 0 para rutas locales")
    ap.add_argument("--vjp-batch", type=int, default=DEFAULT_VJP_BATCH,
                    help="Tamaño de batch para VJP por neurona (candidatos en paralelo)")
    ap.add_argument("--mode", type=str, default=DEFAULT_MODE,
                    choices=["full","prefilter_only","vjp_only"],
                    help="Ejecución: 'full' (prefiltro+VJP), 'prefilter_only' o 'vjp_only'")
    ap.add_argument("--prefilter-out", type=str, default=DEFAULT_PREFILTER_OUT,
                    help="Ruta JSONL donde guardar/leer candidatos del prefiltro")
    args = ap.parse_args()
    # Override models_config without env vars
    try:
        cfg.DALLE_MODEL = args.model_path
        cfg.DALLE_PROCESSOR = args.processor_id
        cfg.USE_SNAPSHOT_DOWNLOAD = bool(args.use_snapshot)
    except Exception:
        pass
    logger.info("Config: dataset=%s out=%s k=%d prefilter_n=%d top_m=%d batch=%d chunk=%d model=%s processor=%s snapshot=%s",
                args.dataset, args.out, args.k, args.prefilter_n, args.top_m_per_image,
                args.batch_size, args.chunk_size, args.model_path, args.processor_id,
                str(bool(args.use_snapshot)))
    run(
        dataset_dir=args.dataset,
        out_path=args.out,
        k=args.k,
        prefilter_n=args.prefilter_n,
        top_m_per_image=args.top_m_per_image,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

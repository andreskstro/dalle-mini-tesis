# generator.py
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
import loader
import logging
from functools import partial
from flax.training.common_utils import shard
from flax.jax_utils import replicate
import models_config as cfg
import uuid


# Cache en memoria: batch_id -> datos necesarios para EB posterior
_BATCH_CACHE: Dict[str, Dict[str, Any]] = {}

# Logger para el progreso de la carga
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(h)

def _fail(msg: str):
    raise RuntimeError(msg)

def _tokens_to_pil(vqgan, vqgan_params, indices: np.ndarray) -> Image.Image:
    # DALL·E mini genera 256 tokens de imagen (16x16). VQGAN suele esperar (B, H, W)
    # Aseguramos int y forma adecuada
    idx = np.asarray(indices).astype(np.int32)
    if idx.ndim == 1 and idx.size == 256:
        idx = idx.reshape(16, 16)
    dec = vqgan.decode_code(idx[None, ...], params=vqgan_params)
    arr = np.asarray(dec[0]).clip(0.0, 1.0)
    arr_u8 = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")

def generate_images(prompt: str, num_images: int = 1) -> List[Image.Image]:
    if num_images < 1:
        return []

    bundle = loader.get_bundle(force_reload=False)
    if not isinstance(bundle, dict):
        _fail("Modelo no cargado (bundle vacío).")
    
    model = bundle.get("model")
    params = bundle.get("params")
    processor = bundle.get("processor")
    vqgan = bundle.get("vqgan")
    vqgan_params = bundle.get("vqgan_params")
    try:
        _logger.info("Bundle keys: %s", list(bundle.keys()))
        _logger.info("Tipos -> model:%s params:%s processor:%s vqgan:%s vqgan_params:%s",
                  type(model).__name__, type(params).__name__, type(processor).__name__,
                  type(vqgan).__name__, type(vqgan_params).__name__)
    except Exception:
        pass

    if model is None or params is None:
        _fail("Modelo/params no disponibles.")
    if processor is None:
        _fail("Processor no cargado.")
    if vqgan is None or vqgan_params is None:
        _fail("VQGAN no cargado.")

    # Tokenización (exacta al notebook): DalleBartProcessor(prompts) sin kwargs
    try:
        n_devices = max(1, jax.device_count())
        prompts = [prompt] * n_devices  # Igualamos batch a n_devices como en el notebook
        tokenized_prompts = processor(prompts)
    except Exception as e:
        _logger.exception("Fallo tokenizando con DalleBartProcessor(prompts)")
        raise RuntimeError(f"Error en tokenización: {type(e).__name__}: {e}") from e

    try:
        # Log liviano de éxito de tokenización
        _logger.info("Tokenización realizada correctamente")
    except Exception:
        pass

    # Asegura que tenga input_ids / attention_mask y tipos int32
    for k in ("input_ids", "attention_mask"):
        if k not in tokenized_prompts:
            _fail(f"Tokenización inválida: falta '{k}'.")
        if not hasattr(tokenized_prompts[k], "shape"):
            tokenized_prompts[k] = np.array(tokenized_prompts[k])
        tokenized_prompts[k] = np.asarray(tokenized_prompts[k]).astype(np.int32, copy=False)

    # Notebook: definir p_generate y p_decode con pmap
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    try:
        _logger.info("Tokenización asegurada: input_ids.shape=%s attention_mask.shape=%s",
                     getattr(tokenized_prompts.get('input_ids', None), 'shape', None),
                     getattr(tokenized_prompts.get('attention_mask', None), 'shape', None))
    except Exception:
        pass

    cond_scale = 10.0
    top_k = None
    top_p = None
    temperature = None
    key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))

    # Preparar sharding/replicación como en el notebook
    n_devices = jax.device_count()
    if n_devices < 1:
        _fail("No hay dispositivos JAX disponibles.")

    # shard de inputs y keys
    tokenized_shard = shard(tokenized_prompts)
    params_repl = replicate(params)
    vqgan_params_repl = replicate(vqgan_params)

    images: List[Image.Image] = []
    # Guardar sequences de cada imagen generada para EB posterior
    all_sequences: List[np.ndarray] = []

    for _ in range(max(1, num_images)):
        # Generar — usa nueva semilla por iteración para obtener imágenes distintas
        try:
            _logger.info("Invocando p_generate (jit/compilación puede tardar la primera vez)...")
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, n_devices)
            out = p_generate(tokenized_shard, keys, params_repl, top_k, top_p, temperature, cond_scale)
            _logger.info("p_generate completado.")
        except Exception as e:
            raise RuntimeError(f"Error en p_generate/model.generate: {type(e).__name__}: {e}") from e

        # Extraer sequences de forma robusta (el retorno de pmap puede ser objeto o dict)
        sequences = None
        try:
            sequences = getattr(out, "sequences", None)
        except Exception:
            sequences = None
        if sequences is None and isinstance(out, dict):
            sequences = out.get("sequences")
        if sequences is None:
            _fail("El modelo no devolvió 'sequences'.")

        seq = np.asarray(sequences)
        try:
            _logger.info("sequences dtype=%s shape=%s", seq.dtype, getattr(seq, 'shape', None))
        except Exception:
            pass
        # Esperado: [n_devices, batch, seq_len]
        while seq.ndim < 3:
            seq = seq[None, ...]
        # Mantener batch como en el notebook, remover BOS y reorganizar a (n_devices, batch, 16, 16)

        encoded_images = seq[..., :256]
        if encoded_images.shape[-1] != 256:
            _fail(f"Secuencia inválida (esperado 256 tokens de imagen, got {encoded_images.shape[-1]}).")
        n_dev, batch, _ = encoded_images.shape
        image_tokens = encoded_images.reshape(n_dev, batch, 16, 16)
        image_tokens = np.asarray(image_tokens).astype(np.int32, copy=False)
        try:
            _logger.info("image_tokens dtype=%s shape=%s", image_tokens.dtype, image_tokens.shape)
        except Exception:
            pass

        # vqgan_jax.decode_code espera (batch, num_tokens). Aplanamos 16x16 -> 256
        image_tokens_flat = image_tokens.reshape(n_dev, batch, 256)
        try:
            _logger.info("image_tokens_flat dtype=%s shape=%s", image_tokens_flat.dtype, image_tokens_flat.shape)
        except Exception:
            pass

        # Decodificar con p_decode (o camino directo en single-device)
        try:
            _logger.info("Invocando p_decode...")
            if n_dev == 1:
                # Camino directo: evitar pmap cuando solo hay 1 device
                # image_tokens_flat[0] -> (batch, 256)
                decoded = vqgan.decode_code(image_tokens_flat[0], params=vqgan_params)
            else:
                # Con pmap, cada réplica recibirá (batch, 256)
                decoded = p_decode(image_tokens_flat, vqgan_params_repl)
            _logger.info("p_decode completado.")
        except Exception as e:
            _logger.exception("Fallo en p_decode/VQGAN")
            raise RuntimeError(f"Error en p_decode/VQGAN: {type(e).__name__}: {e}") from e

        # Tomar imagen y normalizar forma/dtype para PIL (idéntico en espíritu al notebook)
        try:
            dec_np = np.asarray(decoded)
            _logger.info("decoded dtype=%s shape=%s", dec_np.dtype, getattr(dec_np, 'shape', None))
            # Seleccionar primera imagen respetando forma 4D o 5D
            if dec_np.ndim == 5:
                # (n_devices, batch, H, W, C)
                arr = dec_np[0, 0]
            elif dec_np.ndim == 4:
                # (n_devices, H, W, C) — camino single-device directo
                arr = dec_np[0]
            else:
                # Fallback: ya es (H, W, C)
                arr = dec_np
            try:
                _logger.info("selected arr shape=%s dtype=%s", getattr(arr, 'shape', None), getattr(arr, 'dtype', None))
            except Exception:
                pass
            # Si el rango está en [-1, 1], reescalar a [0, 1]
            try:
                mn, mx = float(arr.min()), float(arr.max())
                _logger.info("decoded min=%f max=%f", mn, mx)
                if mn < 0.0 or mx > 1.0:
                    arr = (arr + 1.0) / 2.0
            except Exception:
                pass
            arr = np.clip(arr, 0.0, 1.0)
            arr_u8 = (arr * 255).round().astype(np.uint8)
            _logger.info("final arr_u8 shape=%s dtype=%s", arr_u8.shape, arr_u8.dtype)
            images.append(Image.fromarray(arr_u8, mode="RGB"))
            # Guardar la secuencia de tokens generados (sin BOS) del primer device, cada iter representa una imagen
            try:
                gen_tokens_no_bos = encoded_images[0, 0]  # [256]
                all_sequences.append(np.asarray(gen_tokens_no_bos, dtype=np.int32))
            except Exception:
                # Si no se puede extraer, guardamos vacío para evitar romper el flujo
                all_sequences.append(np.zeros((256,), dtype=np.int32))
        except Exception as e:
            raise RuntimeError(f"Error postprocesando imagen: {type(e).__name__}: {e}") from e

    return images


def generate_images_with_cache(prompt: str, num_images: int = 9) -> Tuple[str, List[str], List[str]]:
    """
    Genera `num_images` imágenes, retorna (batch_id, image_data_urls, image_ids) y
    cachea lo necesario para un análisis EB posterior por imagen.
    """
    if num_images < 1:
        num_images = 9

    # Reusar la lógica de generate_images, pero necesitamos también tokenized_prompts y sequences.
    bundle = loader.get_bundle(force_reload=False)
    model = bundle.get("model")
    params = bundle.get("params")
    processor = bundle.get("processor")
    vqgan = bundle.get("vqgan")
    vqgan_params = bundle.get("vqgan_params")

    if model is None or params is None or processor is None or vqgan is None or vqgan_params is None:
        _fail("Bundle incompleto para generación con cache.")

    # Tokenización
    n_devices = max(1, jax.device_count())
    prompts = [prompt] * n_devices
    tokenized_prompts = processor(prompts)
    for k in ("input_ids", "attention_mask"):
        if k not in tokenized_prompts:
            _fail(f"Tokenización inválida: falta '{k}'.")
        if not hasattr(tokenized_prompts[k], "shape"):
            tokenized_prompts[k] = np.array(tokenized_prompts[k])
        tokenized_prompts[k] = np.asarray(tokenized_prompts[k]).astype(np.int32, copy=False)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    cond_scale = 1.0
    top_k = None
    top_p = 0.93
    temperature = 0.85
    key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))

    n_devices = jax.device_count()
    tokenized_shard = shard(tokenized_prompts)
    params_repl = replicate(params)
    vqgan_params_repl = replicate(vqgan_params)

    from io import BytesIO
    import base64

    def pil_to_data_url(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        import base64 as _b64
        b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")
        return "data:image/png;base64," + b64

    images_data_urls: List[str] = []
    sequences_list: List[np.ndarray] = []

    for _ in range(max(1, num_images)):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_devices)



        out = p_generate(tokenized_shard, keys, params_repl, top_k, top_p, temperature, cond_scale)

        sequences = None
        try:
            sequences = getattr(out, "sequences", None)
        except Exception:
            sequences = None
        if sequences is None and isinstance(out, dict):
            sequences = out.get("sequences")
        if sequences is None:
            _fail("El modelo no devolvió 'sequences'.")

        seq = np.asarray(sequences)
        while seq.ndim < 3:
            seq = seq[None, ...]

        # --- DEBUG ---
        print(f"TOKENS: {seq.flatten()[:10]}")

        encoded_images = seq[..., 1:]  # remove BOS
        if encoded_images.shape[-1] != 256:
            _fail(f"Secuencia inválida (esperado 256 tokens de imagen, got {encoded_images.shape[-1]}).")

        # Cachear secuencia de tokens (sin BOS) por imagen (primer device, primer batch)
        try:
            gen_tokens_no_bos = np.asarray(encoded_images[0, 0], dtype=np.int32)  # [256]
        except Exception:
            gen_tokens_no_bos = np.zeros((256,), dtype=np.int32)
        sequences_list.append(gen_tokens_no_bos)

        # Decodificar a imagen
        n_dev, batch, _ = encoded_images.shape
        image_tokens = encoded_images.reshape(n_dev, batch, 16, 16)
        image_tokens_flat = image_tokens.reshape(n_dev, batch, 256)
        if n_dev == 1:
            decoded = vqgan.decode_code(image_tokens_flat[0], params=vqgan_params)
        else:
            decoded = p_decode(image_tokens_flat, vqgan_params_repl)

        dec_np = np.asarray(decoded)
        if dec_np.ndim == 5:
            arr = dec_np[0, 0]
        elif dec_np.ndim == 4:
            arr = dec_np[0]
        else:
            arr = dec_np
        mn, mx = float(arr.min()), float(arr.max())
        if mn < 0.0 or mx > 1.0:
            arr = (arr + 1.0) / 2.0
        arr = np.clip(arr, 0.0, 1.0)
        arr_u8 = (arr * 255).round().astype(np.uint8)
        images_data_urls.append(pil_to_data_url(Image.fromarray(arr_u8, mode="RGB")))

    # Cachear para EB
    batch_id = uuid.uuid4().hex[:8]
    image_ids = [f"{batch_id}_{i}" for i in range(len(images_data_urls))]
    _BATCH_CACHE[batch_id] = {
        "tokenized_prompts": tokenized_prompts,
        "sequences": sequences_list,  # list of np arrays [256]
        "bundle": bundle,
        "image_ids": image_ids,
    }

    return batch_id, images_data_urls, image_ids


def analyze_cached_image_eb(batch_id: str, image_id: str, k: int = 10) -> Dict[str, Any]:
    print(f"--- INICIO ANALYZE EB (Global) para {image_id} ---")
    try:
        # --- CARGA DE DATOS ---
        if batch_id not in _BATCH_CACHE:
            _fail(f"batch_id desconocido: {batch_id}")

        entry = _BATCH_CACHE[batch_id]
        image_ids: List[str] = entry.get("image_ids", [])
        try:
            idx = image_ids.index(image_id)
        except ValueError:
            _fail(f"image_id {image_id} no encontrado en batch {batch_id}")

        tokenized_prompts = entry["tokenized_prompts"]
        sequences_list = entry["sequences"]
        bundle = entry["bundle"]

        model = bundle.get("model")
        params = bundle.get("params")
        W_lm = bundle.get("W_lm")     # [Hidden, Vocab]
        pinv_W = bundle.get("pinv_W") # [Vocab, Hidden]
        
        if model is None or params is None:
            _fail("Modelo/params no disponibles.")
        if W_lm is None:
            _fail("W_lm no disponible en bundle.")

        import jax.numpy as jnp
        import numpy as np

        # Obtenemos la dimensión esperada del hidden state desde la matriz de pesos
        # W_lm shape es (Hidden, Vocab), por ejemplo (1024, 16385)
        expected_hidden_dim = W_lm.shape[0]
        print(f"Dimensión Hidden esperada: {expected_hidden_dim}")

        # --- 1. PREPARACIÓN DE INPUTS ---
        enc_input_ids = jnp.array(tokenized_prompts["input_ids"][0:1], dtype=jnp.int32)
        enc_attention_mask = jnp.array(tokenized_prompts["attention_mask"][0:1], dtype=jnp.int32)
        pos_enc = jnp.arange(enc_input_ids.shape[1], dtype=jnp.int32)[None, :]

        raw_seq = sequences_list[idx] # np array [256]
        gen_tokens_no_bos = jnp.array(raw_seq, dtype=jnp.int32)[None, :]

        start_id = 0
        if hasattr(model, "config"):
            start_id = getattr(model.config, "decoder_start_token_id", 0)
        
        decoder_input_ids = jnp.concatenate(
            [jnp.array([[start_id]], dtype=jnp.int32), gen_tokens_no_bos[:, :-1]], axis=1
        )
        decoder_attention_mask = jnp.ones_like(decoder_input_ids, dtype=jnp.int32)
        pos_dec = jnp.arange(decoder_input_ids.shape[1], dtype=jnp.int32)[None, :]

        inputs = {
            "input_ids": enc_input_ids,
            "attention_mask": enc_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "position_ids": pos_enc,
            "decoder_position_ids": pos_dec,
        }

        print("Ejecutando Forward Pass...")
        
        # --- 2. FORWARD PASS ---
        y_out_top, inter = model.module.apply(
            {"params": params},
            **inputs,
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        logits_seq = y_out_top.logits # [1, 256, Vocab]

        # --- BÚSQUEDA DEL HIDDEN STATE CORRECTO ---
        h_seq_native = None
        
        def _pick_native_hidden_seq(intermediates, target_dim):
            candidates = []
            try:
                def walk(d):
                    if isinstance(d, (list, tuple)):
                        for x in d: walk(x)
                    elif hasattr(d, "items"):
                        for k, v in d.items(): walk(v)
                    else:
                        try:
                            # Buscamos tensor [..., 256, target_dim]
                            if hasattr(d, 'ndim') and hasattr(d, 'shape'):
                                if d.ndim == 3 and d.shape[1] == 256:
                                    # FILTRO CLAVE: La última dimensión debe coincidir con W_lm (1024)
                                    if d.shape[2] == target_dim:
                                        candidates.append(d)
                        except: pass
                
                target = inter
                if isinstance(inter, tuple) and len(inter) > 0:
                    target = inter[0]
                
                walk(target)
            except Exception as e:
                print(f"Warn: Error caminando intermediates: {e}")
            
            # Devolvemos el último candidato válido encontrado
            return candidates[-1] if candidates else None

        h_seq_native = _pick_native_hidden_seq(inter, expected_hidden_dim)

        if h_seq_native is not None:
            H_all = h_seq_native
            print(f"Hidden nativo encontrado CORRECTO: shape={H_all.shape}")
        else:
            print(f"Hidden nativo de dimensión {expected_hidden_dim} no encontrado. Usando fallback (pinv).")
            if pinv_W is None:
                _fail("pinv_W es None, no se puede reconstruir hidden.")
            # Reconstrucción: [1, 256, Vocab] @ [Vocab, Hidden] -> [1, 256, Hidden]
            H_all = jnp.dot(logits_seq, pinv_W)
            print(f"Hidden reconstruido: shape={H_all.shape}")

        # --- 3. CÁLCULO DE ATRIBUCIÓN ACUMULADA ---
        print("Calculando atribución acumulada...")
        
        H_np = np.asarray(H_all[0])        # [256, Hidden] (debería ser 256, 1024)
        W_np = np.asarray(W_lm)            # [Hidden, Vocab] (1024, 16385)
        target_tokens = np.asarray(raw_seq).astype(int) 

        vocab_size = W_np.shape[1]
        hidden_dim = W_np.shape[0] # 1024
        
        # Verificación final de dimensiones antes de operar
        if H_np.shape[1] != hidden_dim:
            _fail(f"Mismatch crítico: H_np tiene dim {H_np.shape[1]} pero W espera {hidden_dim}")

        W_pos = np.maximum(W_np, 0.0)
        total_attribution = np.zeros(hidden_dim, dtype=np.float32)
        eps = 1e-12

        for t in range(256):
            tid = target_tokens[t]
            if tid < 0 or tid >= vocab_size:
                continue 
            
            h_t = H_np[t]           # [1024]
            w_col = W_pos[:, tid]   # [1024]
            
            # AHORA SÍ: (1024,) * (1024,) -> (1024,)
            raw_contrib = h_t * w_col
            denom = raw_contrib.sum() + eps
            
            total_attribution += (raw_contrib / denom)

        # --- 4. RETORNO TOP-K ---
        idxs = total_attribution.argsort()[-k:][::-1]
        vals = total_attribution[idxs]
        
        print(f"Análisis completado. Top idx: {idxs[:3]}")

        return {
            "eb_top_indices": [int(i) for i in idxs.tolist()],
            "eb_top_values": [float(x) for x in vals.tolist()],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        _fail(f"Error interno en EB Analysis: {str(e)}")
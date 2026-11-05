# loader.py
"""
Loader robusto para DALLE-mini (mini) y VQGAN.
Carga modelo + params + processor + vqgan (si disponible).
Soporta snapshot_download (HF) y carga local.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import models_config as cfg

# Logger para el progreso de la carga
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(h)

_BUNDLE: Optional[Dict] = None

def _ensure_cache_dir() -> str:
    p = Path(cfg.HF_CACHE_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _logger.warning("No se pudo crear HF_CACHE_DIR %s: %s", p, e)
    return str(p)

def _snapshot_download(repo_id: str, cache_dir: str) -> Optional[str]:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        _logger.info("huggingface_hub no disponible; skip snapshot_download.")
        return None

    try:
        _logger.info("snapshot_download: %s -> %s", repo_id, cache_dir)
        local_dir = snapshot_download(repo_id, cache_dir=cache_dir, resume_download=True)
        _logger.info("Snapshot en %s", local_dir)
        return local_dir
    except Exception as e:
        _logger.warning("snapshot_download falló: %s", e)
        return None

def _load_via_dalle_mini(local_repo_path: Optional[str], model_id: str):
    try:
        import jax
        import jax.numpy as jnp
        from dalle_mini import DalleBart, DalleBartProcessor
    except Exception as e:
        # Incluir detalle de la excepción para facilitar el debug de versiones/incompatibilidades
        raise RuntimeError(f"Fallo importando dependencias (jax/dalle_mini): {type(e).__name__}: {e}") from e

    src_model = local_repo_path or model_id
    src_proc = model_id

    _logger.info("Cargando via dalle_mini — model=%s | processor=%s", src_model, src_proc)

    # Processor desde repo (alineado al notebook, sin kwargs adicionales)
    try:
        processor = DalleBartProcessor.from_pretrained(src_proc, _do_init=False)
        tokenizer = None  # No es imprescindible para el flujo actual
    except Exception as e:
        raise RuntimeError(f"Error cargando DalleBartProcessor o DalleBartTokenizer desde {src_proc}: {e}") from e

    # Forzar float32 para evitar problemas de cast/scatter y mejorar compatibilidad en CPU
    try:
        dtype = jnp.float32
    except Exception:
        dtype = None

    try:
        model_and_params = DalleBart.from_pretrained(
            src_model,
            revision=cfg.DALLE_COMMIT_ID or None,
            dtype=dtype,
            _do_init=False
        )
        if isinstance(model_and_params, tuple) and len(model_and_params) == 2:
            model, params = model_and_params
        else:
            model = model_and_params
            params = getattr(model, "params", None)
    except Exception as e:
        raise RuntimeError(f"Error cargando DalleBart desde {src_model}: {e}") from e

    # Validación estricta: no continuar si no hay parámetros del modelo
    if params is None:
        raise RuntimeError(
            "DalleBart cargado sin parámetros (params=None). Verifica el repo/revision o el método de carga."
        )

    vqgan = None
    vqgan_params = None
    try:
        from vqgan_jax.modeling_flax_vqgan import VQModel
        vqgan_and_params = VQModel.from_pretrained(
            cfg.VQGAN_REPO,
            revision=cfg.VQGAN_COMMIT_ID or None,
            _do_init=False
        )
        if isinstance(vqgan_and_params, tuple) and len(vqgan_and_params) == 2:
            vqgan, vqgan_params = vqgan_and_params
        else:
            vqgan = vqgan_and_params
            vqgan_params = getattr(vqgan, "params", None)
    except Exception as e:
        # No continuar en silencio: VQGAN es requerido por generator.py
        raise RuntimeError(f"Error cargando VQGAN ({cfg.VQGAN_REPO}@{cfg.VQGAN_COMMIT_ID or 'latest'}): {e}") from e

    # Validación estricta: no continuar si no hay parámetros de VQGAN
    if vqgan_params is None:
        raise RuntimeError(
            "VQGAN cargado sin parámetros (vqgan_params=None). Verifica el repo/revision o el método de carga."
        )

    # Convertir params a float32 en CPU para evitar cast/artefactos (opcional pero recomendado)
    try:
        from jax.tree_util import tree_map
        def _to_f32(x):
            try:
                import jax.numpy as jnp
                if hasattr(x, "dtype") and getattr(x, "dtype") is not None and x.dtype == jnp.float16:
                    return x.astype(jnp.float32)
            except Exception:
                pass
            return x
        params = tree_map(_to_f32, params)
        vqgan_params = tree_map(_to_f32, vqgan_params)
    except Exception:
        pass

    # Extra: cachear W_lm y su seudoinversa para EB en runtime
    W_lm = None
    pinv_W = None
    try:
        import jax.numpy as jnp  # aseguramos disponibilidad local
        # Esperado: params["lm_head"]["kernel"] con forma [hidden, vocab]
        W_lm = params.get("lm_head", {}).get("kernel", None) if isinstance(params, dict) else None
        if W_lm is None:
            # Intento adicional para FrozenDict
            try:
                from flax.core import FrozenDict
                if isinstance(params, FrozenDict):
                    lm_head = params.get("lm_head")
                    if lm_head is not None:
                        W_lm = lm_head.get("kernel")
            except Exception:
                pass
        if W_lm is not None:
            pinv_W = jnp.linalg.pinv(W_lm)
    except Exception as e:
        _logger.warning("No se pudo preparar W_lm/pinv_W para EB: %s", e)

    return {
        "model": model,
        "params": params,
        "processor": processor,
        "tokenizer": tokenizer,  # Aseguramos que el tokenizer esté disponible
        "backend": "dalle-mini",
        "local_repo_path": local_repo_path,
        "cache_dir": None,
        "vqgan": vqgan,
        "vqgan_params": vqgan_params,
        # Campos para EB (pueden ser None si no se hallan)
        "W_lm": W_lm,
        "pinv_W": pinv_W,
    }

def _load_via_transformers_flax(local_repo_path: Optional[str], model_id: str):
    """
    Fallback no implementado en este proyecto. Se define para evitar NameError y
    dar un mensaje claro si se intenta esta ruta.
    """
    raise RuntimeError(
        "Ruta 'transformers+flax' no implementada. Usa la ruta 'dalle-mini' con JAX/Flax "
        "o implementa este loader si realmente la necesitas."
    )

def get_bundle(force_reload: bool = False, model_id: Optional[str] = None) -> Dict:
    global _BUNDLE
    if _BUNDLE is not None and not force_reload:
        return _BUNDLE

    model_id = model_id or getattr(cfg, "DALLE_MODEL", None)
    if not model_id:
        raise RuntimeError("No se encontró DALLE_MODEL en models_config.")

    cache_dir = _ensure_cache_dir()
    local_repo = None
    if cfg.USE_SNAPSHOT_DOWNLOAD:
        try:
            local_repo = _snapshot_download(model_id, cache_dir)
        except Exception:
            local_repo = None

    errors = []
    try:
        bundle = _load_via_dalle_mini(local_repo, model_id)
        bundle["cache_dir"] = cache_dir
        _BUNDLE = bundle
        _logger.info("Bundle cargado vía dalle-mini.")
        return _BUNDLE
    except Exception as e:
        errors.append(("dalle-mini", str(e)))
        _logger.debug("dalle-mini fallo: %s", e)

    try:    
        bundle = _load_via_transformers_flax(local_repo, model_id)
        bundle["cache_dir"] = cache_dir
        _BUNDLE = bundle
        _logger.info("Bundle cargado vía transformers+flax.")
        return _BUNDLE
    except Exception as e:
        errors.append(("transformers-flax", str(e)))
        _logger.debug("transformers+flax fallo: %s", e)

    missing = {"dalle-mini", "transformers", "flax", "jax", "huggingface_hub"}
    install_cmd = "pip install " + " ".join(sorted(missing))
    err_text = (
        "No se pudo cargar el modelo. Intentos y errores:\n"
        + "\n".join([f" - {k}: {v}" for k, v in errors])
        + "\n\nPaquetes sugeridos a instalar:\n  "
        + install_cmd
        + "\n\nSugerencias:\n"
        + "  • Prueba con USE_SNAPSHOT_DOWNLOAD=0 para cargar directo desde el Hub.\n"
        + "  • Asegúrate de tener 'git+https://github.com/borisdayma/dalle-mini.git' y 'git+https://github.com/patil-suraj/vqgan-jax.git' instalados.\n"
        + "  • Si JAX no detecta GPU, funcionará en CPU (más lento).\n"
    )
    raise RuntimeError(err_text)


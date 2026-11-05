from __future__ import annotations
import os
from pathlib import Path

def _ensure_dir(p: str | Path) -> str:
    p = Path(p).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def _str2bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


HF_HOME = _ensure_dir(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", _ensure_dir(Path(HF_HOME) / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", _ensure_dir(Path(HF_HOME) / "datasets"))

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", _ensure_dir(Path(HF_HOME) / "hub"))
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")



JAX_PLATFORM = os.environ.get("JAX_PLATFORM", "gpu").strip().lower()  # "gpu"|"cpu"|"tpu"
os.environ.setdefault("JAX_PLATFORM_NAME", JAX_PLATFORM)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"))


DEBUG = _str2bool(os.environ.get("BACKEND_DEBUG", "1"))
USE_SNAPSHOT_DOWNLOAD = _str2bool(os.environ.get("USE_SNAPSHOT_DOWNLOAD", "1"))

os.environ.setdefault("WANDB_DISABLED", "true")

DALLE_MODEL = os.environ.get("DALLE_MODEL", "/home/kstro/proyectos/dalle-mini/dalle-mini-main/outputs/dalle-model-768-recaptioned") 
DALLE_COMMIT_ID = os.environ.get("DALLE_COMMIT_ID")  #

VQGAN_REPO = os.environ.get("VQGAN_REPO", "dalle-mini/vqgan_imagenet_f16_16384")

VQGAN_COMMIT_ID = os.environ.get("VQGAN_COMMIT_ID", "e93a26e7707683d349bf5d5c41c5b0ef69b677a9")
# app.py — FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import json
from io import BytesIO
from PIL import Image
import base64

from generator import generate_images_with_cache, analyze_cached_image_eb
import torch
import open_clip

app = FastAPI(title="DALL·E Mini (Flax) API — OneDrive/Loader")

# Ajusta orígenes según tu front (localhost:5173 si usas Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # p.ej. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Offline TopK integration
# ----------------------------
# Preferir rutas proporcionadas por el usuario, con fallbacks WSL/Windows
_env_jsonl = os.environ.get("TOPK_JSONL_PATH")
_user_jsonl_win_unc = r"\\\\wsl.localhost\\Ubuntu\\home\\kstro\\proyectos\\TesisBack\\tools\\analyze\\offline_topk_lm_head.jsonl"
_user_jsonl_linux = "/home/kstro/proyectos/TesisBack/tools/analyze/offline_topk_lm_head.jsonl"
_default_jsonl_rel = "tools/analyze/offline_topk_lm_head.jsonl"
TOPK_JSONL_PATH = next((p for p in [_env_jsonl, _user_jsonl_win_unc, _user_jsonl_linux, _default_jsonl_rel] if p and os.path.exists(p)), _default_jsonl_rel)

# Resolver IMAGES_ROOT (train_data) con preferencia a la ruta indicada por el usuario
_env_images = os.environ.get("IMAGES_ROOT")
_user_images_win = r"C:\\Users\\Andrés Castro\\Documents\\GitHub\\Tesis\\tesisfront\\train_data"
_user_images_win_ds = r"C:\\Users\\Andrés Castro\\Documents\\GitHub\\Tesis\\tesisfront\\dataset-35k-pokemon"
_user_images_wsl = "/mnt/c/Users/Andrés Castro/Documents/GitHub/Tesis/tesisfront/train_data"
_user_images_wsl_ds = "/mnt/c/Users/Andrés Castro/Documents/GitHub/Tesis/tesisfront/dataset-35k-pokemon"
# User-provided preferred sample_images (UNC to WSL and native Linux)
_user_images_win_unc_ds = r"\\\\wsl.localhost\\Ubuntu\\home\\kstro\\proyectos\\TesisBack\\sample_images"
_user_images_linux_ds = "/home/kstro/proyectos/TesisBack/sample_images"
_here = os.path.abspath(os.path.dirname(__file__))
_candidates = [
    _env_images,
    _user_images_win_unc_ds,
    _user_images_linux_ds,
    _user_images_win,
    _user_images_wsl,
    _user_images_win_ds,
    _user_images_wsl_ds,
    os.path.abspath(os.path.join(_here, "..", "tesisfront", "train_data")),
    os.path.abspath(os.path.join(_here, "..", "..", "tesisfront", "train_data")),
    os.path.abspath(os.path.join(_here, "..", "tesisfront", "dataset-35k-pokemon")),
    os.path.abspath(os.path.join(_here, "..", "..", "tesisfront", "dataset-35k-pokemon")),
]
IMAGES_ROOT = "/home/kstro/proyectos/TesisBack/tools/dataset/dataset-subset-pokemon"
DEFAULT_K = int(os.environ.get("DEFAULT_K", "5"))

_idx: Dict[int, List[Dict[str, Any]]] = {}

def load_topk_jsonl(path: str) -> Dict[int, List[Dict[str, Any]]]:
    idx: Dict[int, List[Dict[str, Any]]] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe JSONL con topk: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            j = int(rec.get("neuron"))
            lst = rec.get("topk", [])
            idx[j] = lst
    return idx

try:
    _idx = load_topk_jsonl(TOPK_JSONL_PATH)
except Exception as e:
    _idx = {}
    print(f"[WARN] No se pudo cargar {TOPK_JSONL_PATH}: {e}")

# Montar imágenes si existe el directorio
SAMPLE_ROOT = "/home/kstro/proyectos/TesisBack/sample_images"
if os.path.isdir(SAMPLE_ROOT):
    app.mount("/img", StaticFiles(directory=SAMPLE_ROOT), name="images")
else:
    print(f"[WARN] IMAGES_ROOT no existe: {SAMPLE_ROOT}. Configura IMAGES_ROOT para servir /img/")

# Resolvedor dinámico por filename sin extensión: /image/{filename}
_IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
def _resolve_image_path(basename: str) -> str | None:
    # intenta plano: basename + each ext
    for ext in _IMG_EXTS:
        p = os.path.join(IMAGES_ROOT, f"{basename}{ext}") if IMAGES_ROOT else None
        if p and os.path.isfile(p):
            return p
    # intenta subcarpetas directas (un nivel)
    if IMAGES_ROOT and os.path.isdir(IMAGES_ROOT):
        try:
            for name in os.listdir(IMAGES_ROOT):
                d = os.path.join(IMAGES_ROOT, name)
                if os.path.isdir(d):
                    for ext in _IMG_EXTS:
                        p = os.path.join(d, f"{basename}{ext}")
                        if os.path.isfile(p):
                            return p
        except Exception:
            pass
    return None

@app.get("/dataset/first")
def dataset_first(n: int = 5):
    try:
        if not SAMPLE_ROOT or not os.path.isdir(SAMPLE_ROOT):
            raise HTTPException(status_code=404, detail="SAMPLE_ROOT not found")
        files = []
        # collect jpg/png/webp in SAMPLE_ROOT and first-level subfolders
        for name in os.listdir(SAMPLE_ROOT):
            p = os.path.join(SAMPLE_ROOT, name)
            if os.path.isfile(p) and any(name.lower().endswith(ext) for ext in _IMG_EXTS):
                files.append(p)
            elif os.path.isdir(p):
                try:
                    for sub in os.listdir(p):
                        sp = os.path.join(p, sub)
                        if os.path.isfile(sp) and any(sub.lower().endswith(ext) for ext in _IMG_EXTS):
                            files.append(sp)
                except Exception:
                    pass
        files.sort()
        out = []
        for p in files[: max(1, min(n, 50))]:
            base = os.path.splitext(os.path.basename(p))[0]
            # try read sibling caption .txt
            caption = None
            try:
                txt_path = os.path.join(os.path.dirname(p), base + ".txt")
                if os.path.isfile(txt_path):
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as tf:
                        caption = tf.read().strip()
            except Exception:
                caption = None
            # prefer jpg url as mounted; but we don't know original ext; build by actual ext
            rel = None
            # try to compute relative under SAMPLE_ROOT for /img
            try:
                rel = os.path.relpath(p, SAMPLE_ROOT).replace("\\", "/")
            except Exception:
                rel = os.path.basename(p)
            out.append({
                "filename": base,
                "image_url": f"/img/{rel}",
                "caption": caption,
            })
        return {"items": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/first")
def dataset_first_alias(n: int = 5):
    return dataset_first(n)


@app.get("/image/{filename}")
def get_image_dynamic(filename: str):
    p = _resolve_image_path(filename)
    if not p:
        return Response(status_code=404)
    # infer mime
    if p.lower().endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif p.lower().endswith(".png"):
        media_type = "image/png"
    elif p.lower().endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "application/octet-stream"
    return FileResponse(p, media_type=media_type)

class GenerateRequest(BaseModel):
    prompt: str
    n_predictions: int = Field(9, ge=1, le=9)

class GenerateResponse(BaseModel):
    batch_id: str
    images: List[str]  # data:image/png;base64,... (always 9 here)
    image_ids: List[str]  # stable ids per image within this batch

def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64

_clip_model = None
_clip_preprocess = None
_clip_device = None

def _get_clip(device: str | None = None):
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev
        )
        _clip_model = model.eval()
        _clip_preprocess = preprocess
        _clip_device = dev
    return _clip_model, _clip_preprocess, _clip_device

def _data_url_to_pil(data_url: str) -> Image.Image:
    if data_url.startswith("data:"):
        base64_str = data_url.split(",", 1)[1]
        data = base64.b64decode(base64_str)
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(data_url).convert("RGB")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate 9 images and return batch_id + image_ids for later EB analysis."""
    try:
        # Always generate 9; ignore n_predictions beyond bounds
        num = min(max(req.n_predictions, 1), 9)
        batch_id, images_data_urls, image_ids = generate_images_with_cache(req.prompt, num_images=num)
        return GenerateResponse(batch_id=batch_id, images=images_data_urls, image_ids=image_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Endpoints: TopK por neurona y agregación
# ----------------------------
class AggregateRequest(BaseModel):
    neurons: List[int]
    weights: List[float] | None = None
    k: int | None = None


@app.get("/api/health")
def topk_health():
    return {"ok": True, "neurons_indexed": len(_idx)}


@app.get("/api/neuron/{neuron_id}/topk")
def get_neuron_topk(neuron_id: int, k: int | None = None):
    # carga perezosa si no se pudo al inicio
    global _idx
    if not _idx:
        try:
            _idx = load_topk_jsonl(TOPK_JSONL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No se pudo cargar topk jsonl: {e}")
    lst = _idx.get(neuron_id)
    if lst is None:
        raise HTTPException(status_code=404, detail=f"neurona {neuron_id} no encontrada")
    kk = k or DEFAULT_K
    return {"neuron": neuron_id, "topk": lst[:kk]}


@app.post("/api/aggregate")
def aggregate(req: AggregateRequest):
    global _idx
    if not _idx:
        try:
            _idx = load_topk_jsonl(TOPK_JSONL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No se pudo cargar topk jsonl: {e}")
    neurons = req.neurons or []
    if not neurons:
        return {"neurons": [], "k": req.k or DEFAULT_K, "results": []}
    weights = req.weights or [1.0] * len(neurons)
    if len(weights) != len(neurons):
        raise HTTPException(status_code=400, detail="weights debe tener mismo largo que neurons")
    s = sum(weights)
    if s > 0:
        weights = [w / s for w in weights]

    acc: Dict[str, Dict[str, Any]] = {}
    for j, w in zip(neurons, weights):
        for it in _idx.get(int(j), []):
            fn = str(it.get("filename"))
            sc = float(it.get("score", 0.0))
            if fn not in acc:
                acc[fn] = {"filename": fn, "score": 0.0, "hits": 0, "caption": it.get("caption")}
            acc[fn]["score"] += w * sc
            acc[fn]["hits"] += 1

    items = list(acc.values())
    items.sort(key=lambda x: x["score"], reverse=True)
    kk = req.k or DEFAULT_K
    out = []
    for it in items[:kk]:
        out.append({
            "filename": it["filename"],
            "score": it["score"],
            "hits": it["hits"],
            "caption": it.get("caption"),
            "image_url": f"/image/{it['filename']}",
        })
    return {"neurons": neurons, "k": kk, "results": out}

# Aliases sin prefijo /api para compatibilidad con proxies que lo eliminan
@app.get("/neuron/{neuron_id}/topk")
def get_neuron_topk_alias(neuron_id: int, k: int | None = None):
    return get_neuron_topk(neuron_id, k)

@app.post("/aggregate")
def aggregate_alias(req: AggregateRequest):
    return aggregate(req)


class AnalyzeRequest(BaseModel):
    batch_id: str
    image_id: str
    k: int = Field(10, ge=1, le=1024)


class AnalyzeResponse(BaseModel):
    eb_top_indices: List[int]
    eb_top_values: List[float]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Run EB on a selected image from a previously generated batch and return EB top-k."""
    try:
        result = analyze_cached_image_eb(req.batch_id, req.image_id, req.k)
        return AnalyzeResponse(
            eb_top_indices=result["eb_top_indices"],
            eb_top_values=result["eb_top_values"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CompareRequest(BaseModel):
    dataset_filename: str
    generated_image_data_url: str
    method: str | None = "clip"


class CompareResponse(BaseModel):
    sim_clip_01: float
    cosine: float
    model: str
    device: str


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    try:
        p = _resolve_image_path(req.dataset_filename)
        if not p:
            raise HTTPException(status_code=404, detail="dataset image not found")
        img_dataset = Image.open(p).convert("RGB")
        img_generated = _data_url_to_pil(req.generated_image_data_url)

        model, preprocess, device = _get_clip()

        with torch.no_grad():
            x = preprocess(img_dataset).unsqueeze(0).to(device)
            y = preprocess(img_generated).unsqueeze(0).to(device)
            fx = torch.nn.functional.normalize(model.encode_image(x).float(), dim=1)
            fy = torch.nn.functional.normalize(model.encode_image(y).float(), dim=1)
            cos = float((fx @ fy.T).item())
            sim01 = (cos + 1.0) / 2.0

        return CompareResponse(
            sim_clip_01=sim01,
            cosine=cos,
            model="ViT-B-32/laion2b_s34b_b79k",
            device=str(device),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

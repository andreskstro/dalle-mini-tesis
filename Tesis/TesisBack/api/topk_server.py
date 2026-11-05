import os
import json
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Config
TOPK_JSONL_PATH = os.environ.get("TOPK_JSONL_PATH", "tools/analyze/offline_topk_lm_head.jsonl")
IMAGES_ROOT = os.environ.get("IMAGES_ROOT", "tools/dataset/images")  # ajusta a tu ruta real
DEFAULT_K = int(os.environ.get("DEFAULT_K", "5"))

# Carga en memoria
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
    # Dejar vacío y reportar en primer request
    _idx = {}
    print(f"[WARN] No se pudo cargar {TOPK_JSONL_PATH}: {e}")

app = FastAPI(title="Neuron TopK API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir imágenes estáticas si el directorio existe
if os.path.isdir(IMAGES_ROOT):
    app.mount("/img", StaticFiles(directory=IMAGES_ROOT), name="images")
else:
    print(f"[WARN] IMAGES_ROOT no existe: {IMAGES_ROOT}. Configura IMAGES_ROOT para servir /img/")

class AggregateRequest(BaseModel):
    neurons: List[int]
    weights: List[float] | None = None
    k: int | None = None

@app.get("/api/health")
def health():
    return {"status": "ok", "neurons_indexed": len(_idx)}

@app.get("/api/neuron/{neuron_id}/topk")
def get_neuron_topk(neuron_id: int, k: int | None = None):
    if not _idx:
        # intenta recargar
        try:
            global _idx
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
    if not _idx:
        try:
            global _idx
            _idx = load_topk_jsonl(TOPK_JSONL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No se pudo cargar topk jsonl: {e}")
    neurons = req.neurons or []
    if not neurons:
        raise HTTPException(status_code=400, detail="neurons vacío")
    weights = req.weights or [1.0] * len(neurons)
    if len(weights) != len(neurons):
        raise HTTPException(status_code=400, detail="weights debe tener mismo largo que neurons")
    # normaliza pesos si es posible
    s = sum(weights)
    if s > 0:
        weights = [w / s for w in weights]

    # suma ponderada de scores por filename
    acc: Dict[str, Dict[str, Any]] = {}
    for j, w in zip(neurons, weights):
        lst = _idx.get(int(j), [])
        for it in lst:
            fn = str(it.get("filename"))
            sc = float(it.get("score", 0.0))
            if fn not in acc:
                acc[fn] = {"filename": fn, "score": 0.0, "caps": [], "hits": 0}
            acc[fn]["score"] += w * sc
            acc[fn]["hits"] += 1
            cap = it.get("caption")
            if cap:
                # guarda hasta una caption de muestra
                if len(acc[fn]["caps"]) < 1:
                    acc[fn]["caps"].append(cap)

    items = list(acc.values())
    items.sort(key=lambda x: x["score"], reverse=True)
    kk = req.k or DEFAULT_K
    # transforma salida
    out = []
    for it in items[:kk]:
        out.append({
            "filename": it["filename"],
            "score": it["score"],
            "hits": it["hits"],
            "caption": it["caps"][0] if it["caps"] else None,
            "image_url": f"/img/{it['filename']}.jpg",
        })
    return {"neurons": neurons, "k": kk, "results": out}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api.topk_server:app", host="0.0.0.0", port=port, reload=False)


Frontend & Backend together 
All code used for our tesis is in this repository

Please take this README as a guide, not as the source of truth, this project has lot of deprecated/difficult dependencies to reach today. Updating this code to newer versions should be considered.

# dalle-mini-tesis

Thesis project: **explaining what a fine-tuned DALL·E Mini model learned from its training data**.

We trained DALL·E Mini on a Pokémon image/caption dataset. A web app lets you type a prompt and generate images. You can then pick one image and ask *which internal neurons were most responsible for it*, using Excitation Backprop (EB). Finally, you can see *which training images most strongly "feed" those neurons*, using gradient-based neuron influence scores.

```
 prompt ──► DALL·E Mini (Flax) + VQGAN ──► 9 generated images
                                               │ pick one
                                               ▼
                               Excitation Backprop on layer L
                                   → top-k neurons J, relevances α
                                               │
                                               ▼
                    precomputed influence index (offline_topk_lm_head.jsonl)
                                   → training images that most influence J
                                               │
                                               ▼
                       CLIP similarity: generated image vs. training image
```

The math behind the influence scores is written up in [`Tesis/neuronal_influence_theory.md`](Tesis/neuronal_influence_theory.md).

---

## Repository layout

| Path | What it is |
|---|---|
| `dalle-mini/dalle-mini-main/` | Our fork of [borisdayma/dalle-mini](https://github.com/borisdayma/dalle-mini), with training configs, the encoded dataset and the trained model checkpoint. |
| `TesisBack/` | **Main backend** (FastAPI). Generates images, runs EB analysis, serves the top-k influence index and CLIP comparisons. This is the most recent version. |
| `Tesis/TesisBack/` | Older copy of the backend. It also has a standalone top-k server (`api/topk_server.py`) and the inference notebook. |
| `Tesis/TesisBackend/` | Early prototype backend (`backend/main.py` returns a dummy image) plus a copy of dalle-mini with the influence experiment scripts. |
| `Tesis/tesisfront/` | **Frontend** (React 19, Vite, TypeScript, Tailwind). |
| `Tesis/dataset/`, `Tesis/*.py` | Dataset preparation scripts: dedupe, clean/rename, caption generation, parquet export. |
| `Tesis/captions.csv` | Captions for the Pokémon dataset (`image,caption`). |

### Inside `dalle-mini/dalle-mini-main/`

| Path | What it is |
|---|---|
| `tools/train/train.py` | Training script (upstream, with our changes). |
| `tools/train/config/mini/config_thesis.json` and others | Model configs we used (`config_tiny`, `config_ft`, `config256_capas`, …). |
| `tools/train/sweep.yaml`, `sweep_exploracion.yaml` | Weights & Biases hyperparameter sweeps. |
| `tools/train/neuron_influence_optimized.py` | Computes per-neuron influence during or after training. |
| `compute_influence.py`, `test_*_influence.py`, `verify_model_health.py` | Influence experiments and sanity checks. |
| `tools/dataset/encode_dataset.ipynb` | Encodes images into VQGAN tokens. |
| `tools/dataset/encoded_data/{train,val}/*.parquet` | Encoded dataset used for training. |
| `outputs/dalle-mini-thesis-hybrid-subset-lowlr/` | Trained checkpoint, stored with **Git LFS** (`flax_model.msgpack` ≈ 774 MB). |

### Inside `TesisBack/`

| Path | What it is |
|---|---|
| `app.py` | FastAPI app with all the endpoints. |
| `generator.py` | Image generation, plus EB analysis of a cached generated image. |
| `loader.py`, `models_config.py` | Load DalleBart, the processor and VQGAN. All settings come from env vars. |
| `tools/analyze/compute_topk_per_neuron.py` | Builds `offline_topk_lm_head.jsonl`: top-k training images for each neuron. |
| `tools/analyze/offline_topk_lm_head.jsonl` | The precomputed influence index the API serves. |
| `tools/dataset/` | Caption generation and token counting. |
| `sample_images/` | A few sample dataset images served at `/img`. |

---

## Data

- **Dataset:** Pokémon images with generated captions, e.g. `Abra, a yellow Psychic Pokémon, is shown as …`.
- `Tesis/tesisfront/dataset-35k-pokemon/` is in the repo (a small subset).
- `Tesis/tesisfront/train_data/` (about 4.3 GB, 71k `.jpg` + `.txt` pairs) is **not in the repo** because it's too big for GitHub. Ask the authors for a copy and put it at that path.
- The VQGAN-encoded version used for training lives in `dalle-mini/dalle-mini-main/tools/dataset/encoded_data/`.

---

## Setup

The model runs on **JAX/Flax**, so the backend is meant to run on **Linux or WSL2 with an NVIDIA GPU**. CPU works too, but it's slow.

### 0. Clone (with LFS)

```bash
git lfs install
git clone https://github.com/andreskstro/dalle-mini-tesis.git
cd dalle-mini-tesis
git lfs pull          # downloads the trained checkpoint
```

### 1. Backend (`TesisBack/`)

```bash
cd TesisBack
python3.10 -m venv venv && source venv/bin/activate

# JAX (pick the build that matches your CUDA, see https://github.com/google/jax#installation)
pip install -U "jax[cuda12]"

# dalle-mini (our fork) + VQGAN
pip install -e ../dalle-mini/dalle-mini-main
pip install git+https://github.com/patil-suraj/vqgan-jax.git

# API dependencies
pip install -r requirements.txt
```

Point the backend at the trained model and the influence index:

```bash
export DALLE_MODEL="$(pwd)/../dalle-mini/dalle-mini-main/outputs/dalle-mini-thesis-hybrid-subset-lowlr"
export TOPK_JSONL_PATH="$(pwd)/tools/analyze/offline_topk_lm_head.jsonl"
export JAX_PLATFORM=gpu        # or cpu
```

Run it:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Check that it's alive: <http://localhost:8000/health> and <http://localhost:8000/docs>.

> ⚠️ `app.py` still has some **hardcoded paths** from our machines (`/home/kstro/proyectos/...`). These are `IMAGES_ROOT`, used to serve dataset images at `/image/{filename}`, and `SAMPLE_ROOT`, served at `/img`. Change them to your local `train_data` (or `dataset-35k-pokemon`) and `sample_images` folders before running.

#### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DALLE_MODEL` | `/home/kstro/.../dalle-model-768-recaptioned` | Path or HF repo of the DalleBart checkpoint. |
| `DALLE_COMMIT_ID` | – | Optional HF revision. |
| `VQGAN_REPO` | `dalle-mini/vqgan_imagenet_f16_16384` | VQGAN decoder. |
| `VQGAN_COMMIT_ID` | `e93a26e…` | VQGAN revision. |
| `JAX_PLATFORM` | `gpu` | `gpu`, `cpu` or `tpu`. |
| `TOPK_JSONL_PATH` | `tools/analyze/offline_topk_lm_head.jsonl` | Precomputed neuron → top-k images index. |
| `DEFAULT_K` | `5` | Default number of results for top-k endpoints. |
| `HF_HOME` | `~/.cache/hf` | Hugging Face cache. |
| `BACKEND_DEBUG` | `1` | Verbose logging. |

### 2. Frontend (`Tesis/tesisfront/`)

Needs Node 20.

```bash
cd Tesis/tesisfront
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to `http://localhost:8000` and strips the `/api` prefix, so the backend must be running on port 8000.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `POST` | `/generate` | `{prompt, n_predictions≤9}` → `{batch_id, images[] (base64 PNG), image_ids[]}`. Results are cached for `/analyze`. |
| `POST` | `/analyze` | `{batch_id, image_id, k}` → EB top-k neuron indices and relevance values for that image. |
| `GET` | `/neuron/{id}/topk?k=` | Training images that most influence neuron `id`. |
| `POST` | `/aggregate` | `{neurons[], weights[]?, k?}` → weighted combination of several neurons' top-k images. |
| `POST` | `/compare` | `{dataset_filename, generated_image_data_url}` → CLIP (ViT-B-32, LAION-2B) similarity. |
| `GET` | `/dataset/first?n=` | First `n` dataset images (for the gallery strip). |
| `GET` | `/image/{filename}` | Serves a dataset image by file name (no extension needed). |

`/api/...` aliases exist for `health`, `neuron/{id}/topk`, `aggregate` and `dataset/first`.

---

## Training (optional)

From `dalle-mini/dalle-mini-main/`:

```bash
# single run
python tools/train/train.py \
  --config_name tools/train/config/mini/config_thesis.json \
  --tokenizer_name dalle-mini/dalle-mini \
  --dataset_repo_or_path parquet \
  --train_file "tools/dataset/encoded_data/train/*.parquet" \
  --validation_file "tools/dataset/encoded_data/val/*.parquet" \
  --output_dir outputs/my-run --do_train --do_eval

# or a W&B sweep
wandb sweep tools/train/sweep.yaml
```

After training, rebuild the influence index with `TesisBack/tools/analyze/compute_topk_per_neuron.py`.

---

## Credits

- Built on [DALL·E Mini](https://github.com/borisdayma/dalle-mini) by Boris Dayma et al. (Apache 2.0, see `dalle-mini/dalle-mini-main/LICENSE`) and [vqgan-jax](https://github.com/patil-suraj/vqgan-jax).
- CLIP similarity uses [OpenCLIP](https://github.com/mlfoundations/open_clip).
- Pokémon images and names belong to their respective owners and are used here for academic research only.

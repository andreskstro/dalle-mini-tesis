from __future__ import annotations

import os
import csv
import argparse
import re
from typing import List

from tqdm import tqdm

# Lazy imports for heavy deps
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


def resize_max_side(img: "Image.Image", max_side: int = 768) -> "Image.Image":
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    try:
        return img.resize((new_w, new_h), resample=Image.BICUBIC)
    except Exception:
        return img.resize((new_w, new_h))


def detect_device() -> str:
    """Best-effort device detection (cuda/mps/cpu)."""
    if torch is not None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


# Patterns to remove boilerplate intros from model outputs
_BOILERPLATE_PATTERNS = [
    # 1. "This is an image..."
    r"^this\s+is\s+(an|the)\s+(animated\s+)?image\.?\s*",
    
    # 2. "In this image, I/we/you can see..." (Ahora todo en minúsculas)
    # Soporta coma opcional, dos puntos, y pronombres i/we/you
    r"^in\s+this\s+(image|picture|photo)[\s,:]*((i|we|you)\s+can\s+)?(see|observe|find|identify)?\s*",
    
    # 3. "Here we can see..."
    r"^here\s+we\s+(can\s+)?(see|observe)\s*",
    
    # 4. "The image shows..."
    r"^the\s+image\s+(shows|displays|depicts|contains|presents)\s*",
    
    # 5. "I can see..." (Directo, sin "In this image")
    r"^(i|we|you)\s+can\s+(see|observe|find|identify)\s*",
    
    # 6. "An image of..."
    r"^an?\s+image\s+of\s*",
]


def polish_caption(txt: str) -> str:
    """Clean a raw caption to be training-friendly.

    - Elimina frases introductorias genéricas ("this is an image", etc.).
    - MANTIENE todas las oraciones (descripción completa).
    - Normaliza espacios y capitalización inicial.
    """
    s = txt.strip()
    low = s.lower()
    
    # 1. Eliminar el boilerplate (frases roboticas)
    for pat in _BOILERPLATE_PATTERNS:
        m = re.match(pat, low)
        if m:
            s = s[m.end() :].lstrip()
            low = s.lower() # actualizar para la siguiente pasada si hay multiples patrones
            
    # 2. ELIMINADO: Ya no cortamos la primera oración.
    # parts = re.split(r"(?<=[.!?])\s+", s)
    # if parts:
    #    s = parts[0].strip()

    # 3. Capitalizar la primera letra del resultado final
    if s:
        s = s[0].upper() + s[1:]
        
    return s if s else txt.strip()


def list_images(dataset_dir: str) -> List[str]:
    """Return list of image paths relative to dataset_dir.

    Recorre recursivamente el directorio y devuelve todas las rutas de
    imágenes soportadas, manteniendo la ruta relativa (para escribirla al CSV).
    """
    rels: List[str] = []
    for root, _, files in os.walk(dataset_dir):
        for fname in sorted(files):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, dataset_dir).replace("\\", "/")
                rels.append(rel_path)
    rels.sort()
    return rels


class Captioner:
    """Wrapper sobre modelos de captioning (Florence-2, GIT, BLIP, etc.)."""

    def __init__(
        self,
        device: str,
        prefer_florence: bool,
        prefer_git: bool,
        batch_size: int,
        florence_model_id: str | None,
        only_florence: bool,
        debug: bool = False,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.pipe = None
        self.florence_proc = None
        self.florence_model = None
        self.model_name = None
        self.only_florence = only_florence
        self.debug = debug
        self._load(prefer_florence, prefer_git, florence_model_id)

    def _load(self, prefer_florence: bool, prefer_git: bool, florence_model_id: str | None) -> None:
        from transformers import pipeline  # type: ignore

        tried: List[str] = []

        def try_load(model_id: str) -> bool:
            nonlocal tried
            tried.append(model_id)
            try:
                device_index = 0 if self.device == "cuda" else -1
                self.pipe = pipeline(
                    task="image-to-text",
                    model=model_id,
                    trust_remote_code=True,
                    device=device_index,
                )
                self.model_name = model_id
                return True
            except Exception:
                return False

        if prefer_florence:
            # Intentar usar Florence-2 nativo primero
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM  # type: ignore

                candidates: List[str] = []
                if florence_model_id:
                    candidates.append(florence_model_id)
                candidates += [
                    "microsoft/Florence-2-large-ft",
                    "microsoft/Florence-2-base-ft",
                    "microsoft/Florence-2-base",
                ]
                use_cuda = torch is not None and self.device == "cuda"
                dtype = torch.float16 if use_cuda else None
                last_err = None
                for model_id in candidates:
                    try:
                        self.florence_proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                        self.florence_model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            trust_remote_code=True,
                            torch_dtype=dtype,
                            device_map=None,
                            attn_implementation="eager",
                            low_cpu_mem_usage=True,
                        )
                        if use_cuda:
                            self.florence_model.to("cuda")
                        self.florence_model.eval()
                        # Alinear pad/eos para evitar decodificaciones vacías
                        try:
                            cfg = self.florence_model.generation_config
                            if getattr(cfg, "pad_token_id", None) is None and getattr(cfg, "eos_token_id", None) is not None:
                                cfg.pad_token_id = cfg.eos_token_id
                        except Exception:
                            pass
                        try:
                            tokenizer = getattr(self.florence_proc, "tokenizer", None)
                            if tokenizer is not None:
                                if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                                    tokenizer.pad_token_id = tokenizer.eos_token_id
                                    tokenizer.pad_token = tokenizer.eos_token
                        except Exception:
                            pass
                        self.model_name = model_id
                        return
                    except Exception as e:  # pragma: no cover - modelo remoto
                        last_err = e
                        self.florence_proc = None
                        self.florence_model = None
                        continue
                if last_err is not None:
                    if self.only_florence:
                        # Último recurso: usar pipeline con Florence
                        try:
                            from transformers import pipeline as _pl  # type: ignore

                            device_index = 0 if self.device == "cuda" else -1
                            for mid in candidates:
                                try:
                                    self.pipe = _pl(
                                        task="image-to-text",
                                        model=mid,
                                        trust_remote_code=True,
                                        device=device_index,
                                    )
                                    self.model_name = mid
                                    return
                                except Exception:
                                    continue
                        except Exception:
                            pass
                        raise last_err
            except Exception:
                # Si falla el camino nativo, intentar pipeline con Florence (si only_florence)
                if self.only_florence:
                    try:
                        from transformers import pipeline as _pl  # type: ignore

                        device_index = 0 if self.device == "cuda" else -1
                        mids: List[str] = []
                        if florence_model_id:
                            mids.append(florence_model_id)
                        mids += [
                            "microsoft/Florence-2-large-ft",
                            "microsoft/Florence-2-base-ft",
                            "microsoft/Florence-2-base",
                        ]
                        for mid in mids:
                            try:
                                self.pipe = _pl(
                                    task="image-to-text",
                                    model=mid,
                                    trust_remote_code=True,
                                    device=device_index,
                                )
                                self.model_name = mid
                                return
                            except Exception:
                                continue
                    except Exception:
                        pass
                    raise
                if try_load("microsoft/Florence-2-large-ft") or try_load("microsoft/Florence-2-base-ft") or try_load("microsoft/Florence-2-base"):
                    return

        # Preferencia opcional por GIT
        if prefer_git and not self.only_florence:
            if try_load("microsoft/git-large-coco"):
                return

        # Fallbacks genéricos
        if not self.only_florence and try_load("Salesforce/blip-image-captioning-base"):
            return
        if not self.only_florence and try_load("microsoft/git-large-coco"):
            return
        if not self.only_florence and try_load("nlpconnect/vit-gpt2-image-captioning"):
            return

        raise RuntimeError(f"Failed to load any captioning model. Tried: {tried}")

    def describe_batch(self, images: List[Image.Image]) -> List[str]:
        """Devuelve una lista de captions (ya pulidas) para un batch de imágenes."""
        # Camino pipeline genérico
        if self.pipe is not None and (self.florence_model is None or self.florence_proc is None):
            try:
                outputs = self.pipe(
                    images,
                    batch_size=self.batch_size,
                    generate_kwargs={
                        "max_new_tokens": 64,
                        "do_sample": True,
                        "top_p": 0.9,
                        "temperature": 0.8,
                        "no_repeat_ngram_size": 3,
                    },
                )
                caps: List[str] = []
                for o in outputs:
                    if isinstance(o, list) and o and isinstance(o[0], dict) and "generated_text" in o[0]:
                        txt = str(o[0]["generated_text"]).strip()
                    elif isinstance(o, dict) and "generated_text" in o:
                        txt = str(o["generated_text"]).strip()
                    else:
                        txt = str(o).strip()
                    txt = txt.replace("</s>", " ").replace("<s>", " ").replace("<pad>", " ")
                    txt = txt.replace("[PAD]", " ").replace("<unk>", " ")
                    txt = re.sub(r"\s+", " ", txt).strip()
                    txt = txt.strip(" .,;:-")
                    txt = polish_caption(txt)
                    caps.append(txt)
                return caps
            except Exception as e:
                if self.only_florence and self.debug:
                    print(f"[Pipeline error] {repr(e)}")
                if self.only_florence:
                    raise
                # caer al camino Florence si existe

        # Camino Florence nativo
        if self.florence_model is not None and self.florence_proc is not None:
            try:
                pure = getattr(self, "pure_mode", False)

                def build_instruction() -> str:
                    # Instrucción genérica, sin asumir dominio (Pokémon u otro)
                    return (
                        "Describe delicately the image. Mention every specific visual detail: "
                        "the exact texture of the skin/surface, the specific colors and patterns, "
                        "the shape of the body parts, and the background elements."
                    )

                captions: List[str] = [""] * len(images)

                def decode_and_clean(out_ids, used_prompt: str) -> List[str]:
                    if pure:
                        txts = self.florence_proc.batch_decode(out_ids, skip_special_tokens=True)
                        return [polish_caption(t.strip()) for t in txts]
                    txts = self.florence_proc.batch_decode(out_ids, skip_special_tokens=False)
                    cleaned: List[str] = []
                    for t in txts:
                        tt = str(t)
                        if used_prompt:
                            tt = tt.replace(used_prompt, " ")
                        tt = tt.replace("</s>", " ").replace("<s>", " ").replace("<pad>", " ")
                        tt = tt.replace("[PAD]", " ").replace("<unk>", " ")
                        tt = re.sub(r"\s+", " ", tt).strip()
                        tt = tt.strip(" .,;:-")
                        for ph in [
                            "prefer concrete words",
                            "do not say this is an image",
                            "do not say in this image",
                        ]:
                            tt = re.sub(re.escape(ph), "", tt, flags=re.IGNORECASE)
                        tt = re.sub(r"\s+", " ", tt).strip().strip(" .,;:-")
                        tt = polish_caption(tt)
                        cleaned.append(tt)
                    return cleaned

                need = [i for i, c in enumerate(captions) if not c]
                batch_imgs = [images[i] for i in need]
                if pure:
                    task_tok = f"<{getattr(self, 'pure_token', 'DETAILED_CAPTION')}>"
                    text_batch = [task_tok] * len(batch_imgs)
                else:
                    prompt = build_instruction()
                    text_batch = [prompt] * len(batch_imgs)

                inputs = self.florence_proc(text=text_batch, images=batch_imgs, return_tensors="pt")
                casted = {}
                for k, v in inputs.items():
                    if hasattr(v, "to"):
                        try:
                            import torch as _torch

                            if hasattr(v, "dtype") and _torch.is_floating_point(v):
                                casted[k] = v.to(self.florence_model.device, dtype=self.florence_model.dtype)
                            else:
                                casted[k] = v.to(self.florence_model.device)
                        except Exception:
                            casted[k] = v.to(self.florence_model.device)
                    else:
                        casted[k] = v
                inputs = casted
                gen_kwargs = dict(
                    max_new_tokens=56,
                    min_new_tokens=10,
                    do_sample=False,
                    num_beams=5,
                    length_penalty=0.9,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                )
                with torch.inference_mode():
                    out = self.florence_model.generate(**inputs, **gen_kwargs)
                partial = decode_and_clean(out, "" if pure else text_batch[0])
                if self.debug:
                    for j, raw in enumerate(partial[:3]):
                        print(f"[Florence raw] {raw}")
                for idx, txt in zip(need, partial):
                    if txt and len(txt) >= 6:
                        captions[idx] = txt

                # Reintento por imagen si algo quedó muy corto/vacío
                for i in range(len(captions)):
                    if captions[i] and len(captions[i]) >= 8:
                        continue
                    if pure:
                        task_tok = f"<{getattr(self, 'pure_token', 'DETAILED_CAPTION')}>"
                        single_inputs = self.florence_proc(text=[task_tok], images=[images[i]], return_tensors="pt")
                    else:
                        inst = build_instruction()
                        single_inputs = self.florence_proc(text=[inst], images=[images[i]], return_tensors="pt")
                    casted_single = {}
                    for k, v in single_inputs.items():
                        if hasattr(v, "to"):
                            try:
                                import torch as _torch

                                if hasattr(v, "dtype") and _torch.is_floating_point(v):
                                    casted_single[k] = v.to(self.florence_model.device, dtype=self.florence_model.dtype)
                                else:
                                    casted_single[k] = v.to(self.florence_model.device)
                            except Exception:
                                casted_single[k] = v.to(self.florence_model.device)
                        else:
                            casted_single[k] = v
                    alt_kwargs = dict(
                        max_new_tokens=80,
                        min_new_tokens=16,
                        do_sample=False,
                        num_beams=4,
                        length_penalty=1.0,
                        no_repeat_ngram_size=3,
                        use_cache=True,
                    )
                    with torch.inference_mode():
                        out2 = self.florence_model.generate(**casted_single, **alt_kwargs)
                    cap2 = decode_and_clean(out2, "" if pure else build_instruction())[0]
                    if self.debug:
                        print(f"[Florence retry beams] {cap2}")
                    captions[i] = cap2

                captions = [c if c else "" for c in captions]
                return captions
            except Exception as e:
                if self.only_florence:
                    if self.debug:
                        print(f"[Florence describe_batch error] {repr(e)}")
                    raise
                if self.debug:
                    print(f"[Florence describe_batch error] {repr(e)}")
                # caer al pipeline si existe

        # Pipeline genérico como último recurso
        assert self.pipe is not None
        try:
            outputs = self.pipe(images, batch_size=self.batch_size)
        except TypeError:
            outputs = self.pipe(images)

        captions: List[str] = []
        for out in outputs:
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
                captions.append(polish_caption(str(out[0]["generated_text"])))
            elif isinstance(out, dict) and "generated_text" in out:
                captions.append(polish_caption(str(out["generated_text"])))
            else:
                captions.append(polish_caption(str(out)))
        return captions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Ruta al directorio raíz del dataset (se recorre recursivamente)")
    parser.add_argument("--out_csv", default="captions.csv", help="Ruta del CSV de salida")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño de batch para captioning")
    parser.add_argument("--prefer_florence", action="store_true", help="Preferir Florence-2 cuando sea posible")
    parser.add_argument("--prefer_git", action="store_true", help="Preferir microsoft/git-large-coco antes de otros fallbacks")
    parser.add_argument("--florence_model", default=None, help="ID de modelo Florence a usar (p.ej., microsoft/Florence-2-large-ft)")
    parser.add_argument("--only_florence", action="store_true", help="Usar solo Florence; no hacer fallback a otros modelos")
    parser.add_argument("--debug_first_n", type=int, default=0, help="Procesar solo los primeros N ejemplos para depuración")
    parser.add_argument("--debug_print_raw", action="store_true", help="Imprimir algunas captions crudas en consola")
    parser.add_argument("--florence_pure", action="store_true", help="Usar Florence en modo puro: solo token de tarea, sin prompt explícito")
    parser.add_argument("--florence_token", default="DETAILED_CAPTION", help="Token de tarea para modo puro")
    args = parser.parse_args()

    dataset_dir = args.dataset
    out_csv = args.out_csv

    if Image is None:
        raise RuntimeError("Pillow es obligatorio. Instala con: pip install pillow")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Directorio de dataset no encontrado: {dataset_dir}")

    # Preparar captioner
    device = detect_device()
    prefer_florence = args.prefer_florence or (device == "cuda")
    captioner = Captioner(
        device=device,
        prefer_florence=prefer_florence,
        prefer_git=args.prefer_git,
        batch_size=max(8, min(args.batch_size, 16)),
        florence_model_id=args.florence_model,
        only_florence=args.only_florence,
        debug=args.debug_print_raw,
    )
    print(f"Using model: {captioner.model_name} on {device}")
    setattr(captioner, "pure_mode", bool(getattr(args, "florence_pure", False)))
    setattr(captioner, "pure_token", str(getattr(args, "florence_token", "DETAILED_CAPTION")))

    # Recolectar imágenes
    rel_paths = list_images(dataset_dir)
    if args.debug_first_n and args.debug_first_n > 0:
        rel_paths = rel_paths[: args.debug_first_n]
    if not rel_paths:
        print("No se encontraron imágenes.")
        return

    # CSV de salida (UTF-8 con BOM para compatibilidad con Excel)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_f = open(out_csv, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(out_f)
    header = ["image_path", "caption"]
    writer.writerow(header)

    errors = 0
    batch_imgs: List[Image.Image] = []
    batch_meta: List[str] = []  # rel_path

    pbar = tqdm(total=len(rel_paths), desc="Captioning", unit="img")

    def flush_batch() -> None:
        nonlocal errors
        if not batch_imgs:
            return
        try:
            descs = captioner.describe_batch(batch_imgs)
        except Exception as e:
            # Si falla el batch, intentar imagen por imagen
            if captioner.debug:
                print(f"[Batch error] {repr(e)}. Falling back to per-image generation…")
            descs = []
            for img in batch_imgs:
                try:
                    d = captioner.describe_batch([img])[0]
                except Exception as ie:
                    d = ""
                    errors += 1
                    if captioner.debug:
                        print(f"[Per-image error] {repr(ie)}")
                descs.append(d)
        for rel_path, desc in zip(batch_meta, descs):
            caption = desc or ""
            writer.writerow([rel_path, caption])
        batch_imgs.clear()
        batch_meta.clear()

    bs = captioner.batch_size

    for rel_path in rel_paths:
        abs_path = os.path.join(dataset_dir, rel_path)
        try:
            img = Image.open(abs_path).convert("RGB")
            img = resize_max_side(img, 768)
        except Exception:
            errors += 1
            pbar.update(1)
            continue
        batch_imgs.append(img)
        batch_meta.append(rel_path)
        if len(batch_imgs) >= bs:
            flush_batch()
            pbar.update(bs)

    # Vaciar resto
    if batch_imgs:
        n = len(batch_imgs)
        flush_batch()
        pbar.update(n)

    out_f.close()
    pbar.close()

    print(f"Done. Wrote {out_csv}. Errors: {errors}")


if __name__ == "__main__":
    main()

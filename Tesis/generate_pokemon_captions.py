from __future__ import annotations

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
import os
import csv
import json
import time
import math
import argparse
import re
import random
from collections import deque
from typing import Dict, List, Tuple, Set

import requests
from tqdm import tqdm

# Lazy imports for heavy deps
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# Lazy optional import for CLIP refine
try:
    import open_clip  # type: ignore
except Exception:
    open_clip = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

# transformers will be imported lazily inside loader

POKEAPI_BASE = "https://pokeapi.co/api/v2"


def detect_device() -> str:
    if torch is not None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        # Apple Silicon fallback
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


_BOILERPLATE_PATTERNS = [
    r"^this\s+is\s+(an|the)\s+(animated\s+)?image\.?\s*",
    r"^in\s+this\s+(image|picture|photo)\s*(we\s+can\s+)?(see|observe)\s*",
    r"^here\s+we\s+(can\s+)?(see|observe)\s*",
]


def polish_caption(txt: str) -> str:
    s = txt.strip()
    low = s.lower()
    changed = False
    for pat in _BOILERPLATE_PATTERNS:
        m = re.match(pat, low)
        if m:
            s = s[m.end():].lstrip()
            low = s.lower()
            changed = True
    # keep only first sentence to be concise
    parts = re.split(r"(?<=[.!?])\s+", s)
    if parts:
        s = parts[0].strip()
    # capitalize first letter
    if s:
        s = s[0].upper() + s[1:]
    return s if s else txt.strip()


def clean_raw(desc: str, pokemon: str) -> str:
    d = desc or ""
    p = pokemon or ""
    d = d.replace("pokemon character", f"{p}")
    d = d.replace("a cartoon fox", f"a cartoon of {p.lower()}")
    d = d.replace("a drawing of a", f"a drawing of the {p.lower()}")
    return d.strip()


_ANIMAL_LEMMAS_EN: Set[str] | None = None
_ANIMAL_REGEX: "re.Pattern[str]" | None = None

def _build_wordnet_animal_lemmas_en() -> Set[str]:
    try:
        import nltk  # type: ignore
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
        except Exception:
            nltk.download("wordnet")
            from nltk.corpus import wordnet as wn  # type: ignore
        try:
            _ = wn.synset('animal.n.01')
        except Exception:
            nltk.download("wordnet")
        root = wn.synset('animal.n.01')
        seen = set([root])
        stack = [root]
        while stack:
            cur = stack.pop()
            for h in cur.hyponyms():
                if h not in seen:
                    seen.add(h)
                    stack.append(h)
        lemmas: Set[str] = set()
        for s in seen:
            for l in s.lemmas():
                name = l.name().replace('_', ' ').lower()
                if any(ch.isdigit() for ch in name):
                    continue
                if len(name) > 30:
                    continue
                lemmas.add(name)
        lemmas.add("animal")
        lemmas.add("dragon")
        return lemmas
    except Exception:
        return {"animal","dragon","cat","dog","fox","bear","wolf","lion","tiger","leopard","cheetah","mouse","rat","rabbit","hare","deer","boar","pig","goat","sheep","cow","horse","zebra","giraffe","elephant","monkey","ape","gorilla","chimp","panda","koala","kangaroo","sloth","otter","weasel","badger","bird","eagle","hawk","falcon","owl","duck","goose","swan","chicken","turkey","penguin","parrot","pigeon","crow","raven","flamingo","lizard","gecko","iguana","chameleon","crocodile","alligator","turtle","tortoise","snake","python","cobra","viper","frog","toad","salamander","newt","fish","shark","ray","eel","salmon","tuna","whale","dolphin","seal","walrus","crab","lobster","shrimp","octopus","squid","jellyfish","ant","bee","wasp","beetle","butterfly","moth","fly","mosquito","spider","scorpion","sea lion"}

def _animal_regex_en() -> "re.Pattern[str]":
    global _ANIMAL_LEMMAS_EN, _ANIMAL_REGEX
    if _ANIMAL_REGEX is not None:
        return _ANIMAL_REGEX
    if _ANIMAL_LEMMAS_EN is None:
        _ANIMAL_LEMMAS_EN = _build_wordnet_animal_lemmas_en()
    terms = sorted(_ANIMAL_LEMMAS_EN, key=lambda s: -len(s.split()))
    pats: List[str] = []
    for t in terms:
        pats.append(r"\b" + re.escape(t).replace(r"\ ", r"\\s+") + r"\b")
    _ANIMAL_REGEX = re.compile("|".join(pats), flags=re.IGNORECASE)
    return _ANIMAL_REGEX

def normalize_species(desc: str, pokemon: str) -> str:
    s = desc or ""
    rx = _animal_regex_en()
    s = rx.sub("creature", s)
    s = re.sub(r"\babra\b", pokemon.lower(), s, flags=re.IGNORECASE)
    return s


# Varied natural-language templates for final captions
TEMPLATES = [
    "{pokemon}, a {type_color}, is shown as {desc}",
    "{pokemon}, the {type_color}, appears as {desc}",
    "A {type_color} named {pokemon} is depicted as {desc}",
    "{pokemon}, the {type_color}, can be seen as {desc}",
]


def generate_caption_from_template(pokemon: str, type_color: str, desc: str) -> str:
    # Cycle through a shuffled queue to avoid repeating the same template consecutively
    global _TEMPLATE_QUEUE
    try:
        _TEMPLATE_QUEUE
    except NameError:
        _TEMPLATE_QUEUE = deque()
    if not _TEMPLATE_QUEUE:
        shuffled = list(TEMPLATES)
        random.shuffle(shuffled)
        _TEMPLATE_QUEUE.extend(shuffled)
    tpl = _TEMPLATE_QUEUE.popleft()
    return tpl.format(pokemon=pokemon, type_color=type_color, desc=desc)

def post_template_fix(text: str, pokemon: str) -> str:
    s = text
    # Fix pattern: "appears/is shown/can be seen as <NP> is <V-ing>" -> remove the extra 'is'
    s = re.sub(r"\b(appears|is shown|can be seen)\s+as\s+((?:a|an|the)\s+[^\.]*?)\s+is\s+(\w+ing\b)", r"\1 as \2 \3", s, flags=re.IGNORECASE)
    # Also fix generic ' as <NP> is <V-ing>'
    s = re.sub(r"\bas\s+((?:a|an|the)\s+[^\.]*?)\s+is\s+(\w+ing\b)", r"as \1 \2", s, flags=re.IGNORECASE)
    # Insert 'is' after 'that' when missing before gerund
    s = re.sub(r"\bthat\s+(sitting|flying|standing|lying|running|attacking|jumping)\b", r"that is \1", s, flags=re.IGNORECASE)
    # laying -> lying (down)
    s = re.sub(r"\blaying\s+down\b", "lying down", s, flags=re.IGNORECASE)
    s = re.sub(r"\blaying\b", "lying", s, flags=re.IGNORECASE)
    # figure of an Abra -> figure of Abra
    s = re.sub(rf"\bfigure\s+of\s+an\s+{re.escape(pokemon)}\b", f"figure of {pokemon}", s, flags=re.IGNORECASE)
    # Normalize pokemon casing
    s = re.sub(rf"\b{re.escape(pokemon)}\b", pokemon, s, flags=re.IGNORECASE)
    # Article before pokemon (heuristic by first letter)
    if pokemon[:1].lower() in "aeiou":
        s = re.sub(rf"\b[aA]\s+{re.escape(pokemon)}\b", f"an {pokemon}", s)
    else:
        s = re.sub(rf"\b[aA]n\s+{re.escape(pokemon)}\b", f"a {pokemon}", s)
    # Uniform species term in final caption
    s = re.sub(r"\bcreature\b", pokemon, s, flags=re.IGNORECASE)
    # Replace non-informative 'character' by Abra when used as generic species
    s = re.sub(r"\b(cartoon|animated)\s+character\b", rf"\1 {pokemon}", s, flags=re.IGNORECASE)
    # Collapse repeated 'Abra' mentions: keep the first
    s = re.sub(rf"\b({re.escape(pokemon)})(?:[^A-Za-z]+{re.escape(pokemon)})+\b", r"\1", s)
    # Simplify 'Abra <noun> of Abra' -> '<noun> of Abra' (optionally keep leading adjective)
    s = re.sub(rf"\b([A-Za-z]+\s+)?{re.escape(pokemon)}\s+(figure|drawing|picture|statue|toy|model)\s+of\s+{re.escape(pokemon)}\b", r"\1\2 of " + pokemon, s, flags=re.IGNORECASE)
    # Remove duplicated adjacent words (e.g., 'with with')
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # Collapse spaces before punctuation
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_pokemon_name(name: str) -> str:
    s = name.strip().lower()
    # Basic normalizations for known punctuation
    s = s.replace(" ", "-")
    s = s.replace(".", "")
    s = s.replace("'", "")
    s = s.replace("♀", "-f")
    s = s.replace("♂", "-m")
    s = s.replace(":", "-")
    return s


def fetch_pokemon_info(name: str) -> Tuple[List[str], str]:
    """Return (types, color) using PokéAPI. Best-effort and resilient to errors."""
    norm = normalize_pokemon_name(name)
    types: List[str] = []
    color = ""

    # types from /pokemon/{name}
    try:
        r = requests.get(f"{POKEAPI_BASE}/pokemon/{norm}", timeout=15)
        if r.status_code == 200:
            data = r.json()
            t = data.get("types", [])
            # order by slot
            t_sorted = sorted(t, key=lambda x: x.get("slot", 0))
            types = [tt.get("type", {}).get("name", "").title() for tt in t_sorted if tt.get("type")]
        # 404 for forms or alternate names: try species list search as fallback
        elif r.status_code == 404:
            # no-op; we'll rely on species color below
            pass
    except Exception:
        pass

    # color from /pokemon-species/{name}
    try:
        r2 = requests.get(f"{POKEAPI_BASE}/pokemon-species/{norm}", timeout=15)
        if r2.status_code == 200:
            data2 = r2.json()
            color = (data2.get("color", {}) or {}).get("name", "")
            # Map some pokedex color names to common words if needed
            # e.g., 'brown', 'yellow', 'red', etc. Keep as-is.
        # else ignore
    except Exception:
        pass

    return types, color


def load_or_build_types_json(dataset_dir: str, json_path: str) -> Dict[str, Dict[str, object]]:
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Build from folder names
    entries: Dict[str, Dict[str, object]] = {}
    subfolders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    print(f"Building pokemon_types.json for {len(subfolders)} folders via PokéAPI...")
    for name in tqdm(subfolders, desc="PokéAPI", unit="pokemon"):
        types, color = fetch_pokemon_info(name)
        entries[name] = {"types": types, "color": color}
        # be nice to the public API
        time.sleep(0.05)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return entries


def list_images(dataset_dir: str) -> List[Tuple[str, str]]:
    """Return list of (rel_path, folder_name). rel_path is relative to dataset_dir."""
    rels: List[Tuple[str, str]] = []
    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                rels.append((os.path.join(folder, fname).replace("\\", "/"), folder))
    return rels


class Captioner:
    def __init__(self, device: str, prefer_florence: bool, prefer_git: bool, batch_size: int, florence_model_id: str | None, only_florence: bool, debug: bool = False, clip_refine: bool = False) -> None:
        self.device = device
        self.batch_size = batch_size
        self.pipe = None
        self.florence_proc = None
        self.florence_model = None
        self.model_name = None
        self.only_florence = only_florence
        self.debug = debug
        self.clip_refine = clip_refine
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._load(prefer_florence, prefer_git, florence_model_id)

    def _ensure_clip(self):
        if getattr(self, "_clip_model", None) is not None:
            return True
        if open_clip is None or torch is None:
            return False
        try:
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=self.device)
            tokenizer = open_clip.get_tokenizer("ViT-L-14")
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = tokenizer
            return True
        except Exception as e:
            if self.debug:
                print(f"[CLIP load failed] {repr(e)}")
            return False

    def clip_refine_desc(self, img: Image.Image, desc: str, pokemon: str) -> str:
        # Minimal, safe refinement using CLIP text prompts
        try:
            if not self.clip_refine:
                return desc
            if not self._ensure_clip():
                return desc
            candidates = [
                f"a picture of {pokemon}",
                f"a drawing of {pokemon}",
                f"{pokemon} sitting",
                f"{pokemon} standing",
                f"{pokemon} sleeping",
                f"{pokemon} flying",
                f"{pokemon} attacking",
            ]
            image = self._clip_preprocess(img).unsqueeze(0)
            if self.device != "cpu" and hasattr(torch, "device"):
                image = image.to(self.device)
            with torch.no_grad():
                img_feat = self._clip_model.encode_image(image)
                text_tokens = self._clip_tokenizer(candidates)
                if self.device != "cpu" and hasattr(torch, "device"):
                    text_tokens = text_tokens.to(self.device)
                txt_feat = self._clip_model.encode_text(text_tokens)
                # cosine similarities approximated via dot after model's internal normalization
                scores = (img_feat @ txt_feat.T)
                top_idx = int(scores.argmax().item())
                best = candidates[top_idx].lower()
            # Apply light corrections based on best
            out = desc or ""
            # Remove hallucinated objects known to appear
            out = re.sub(r"\b(banana|knife)s?\b", "", out, flags=re.IGNORECASE)
            # Remove dangling connectors after object removal
            out = re.sub(r"\b(with|holding)\s+(a|an)\s*(?:[\.,]|$)", "", out, flags=re.IGNORECASE)
            out = re.sub(r"\b(with|holding)\s+(a|an)\s*$", "", out, flags=re.IGNORECASE)
            # Normalize species to pokemon name rather than generic cat/fox/toy/animal
            try:
                rx_animals = _animal_regex_en()
                out = rx_animals.sub(pokemon.lower(), out)
            except Exception:
                out = re.sub(r"\b(cat|fox|animal)\b", pokemon.lower(), out, flags=re.IGNORECASE)
            out = re.sub(r"\btoy\b", pokemon.lower(), out, flags=re.IGNORECASE)
            # Normalize 'abra' casing and article
            out = re.sub(r"\babra\b", "Abra", out, flags=re.IGNORECASE)
            out = re.sub(r"\ba\s+Abra\b", "an Abra", out, flags=re.IGNORECASE)
            # If best suggests an action, gently steer the verb
            actions = ["sitting", "standing", "sleeping", "flying", "attacking"]
            for act in actions:
                if act in best and not re.search(fr"\b{act}\b", out, flags=re.IGNORECASE):
                    if re.search(r"\bis\b\s+\w+ing", out, flags=re.IGNORECASE):
                        def _rep(m, act=act):
                            return m.group(1) + act
                        out = re.sub(r"(\bis\b\s+)\w+ing", _rep, out, flags=re.IGNORECASE)
                    elif re.search(r"\bis\b", out, flags=re.IGNORECASE):
                        out = re.sub(r"\bis\b", f"is {act}", out, count=1, flags=re.IGNORECASE)
            # Final tidy
            out = re.sub(r"\s+", " ", out).strip()
            out = re.sub(r"\s+\.", ".", out)
            out = out.strip().strip(" .") + "."
            return out
        except Exception as e:
            if self.debug:
                print(f"[CLIP refine error] {repr(e)}")
            return desc

    def _load(self, prefer_florence: bool, prefer_git: bool, florence_model_id: str | None) -> None:
        from transformers import pipeline  # type: ignore

        tried: List[str] = []

        def try_load(model_id: str) -> bool:
            nonlocal tried
            tried.append(model_id)
            try:
                # use device index 0 if cuda, else -1
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
            # Try native Florence-2 API first (more reliable than generic pipeline)
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM  # type: ignore
                candidates = []
                if florence_model_id:
                    candidates.append(florence_model_id)
                candidates += [
                    "microsoft/Florence-2-large-ft",
                    "microsoft/Florence-2-base-ft",
                    "microsoft/Florence-2-base",
                ]
                use_cuda = (torch is not None and self.device == "cuda")
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
                        # Align pad/eos to avoid empty decodes and warnings
                        try:
                            cfg = self.florence_model.generation_config
                            if getattr(cfg, 'pad_token_id', None) is None and getattr(cfg, 'eos_token_id', None) is not None:
                                cfg.pad_token_id = cfg.eos_token_id
                        except Exception:
                            pass
                        try:
                            tokenizer = getattr(self.florence_proc, 'tokenizer', None)
                            if tokenizer is not None:
                                if getattr(tokenizer, 'pad_token_id', None) is None and getattr(tokenizer, 'eos_token_id', None) is not None:
                                    tokenizer.pad_token_id = tokenizer.eos_token_id
                                    tokenizer.pad_token = tokenizer.eos_token
                        except Exception:
                            pass
                        self.model_name = model_id
                        return
                    except Exception as e:
                        last_err = e
                        self.florence_proc = None
                        self.florence_model = None
                        continue
                if last_err is not None:
                    if self.only_florence:
                        # As a last resort under only_florence, try pipeline with Florence
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
                # As a secondary attempt, try the generic pipeline with Florence
                if self.only_florence:
                    # Try pipeline with Florence under only_florence
                    try:
                        from transformers import pipeline as _pl  # type: ignore
                        device_index = 0 if self.device == "cuda" else -1
                        mids = []
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
        # Optional preference: try GIT first
        if prefer_git and not self.only_florence:
            if try_load("microsoft/git-large-coco"):
                return
        # Fallback to BLIP (robust, supports pipeline)
        if not self.only_florence and try_load("Salesforce/blip-image-captioning-base"):
            return
        # Secondary fallback
        if not self.only_florence and try_load("microsoft/git-large-coco"):
            return
        if not self.only_florence and try_load("nlpconnect/vit-gpt2-image-captioning"):
            return
        raise RuntimeError(f"Failed to load any captioning model. Tried: {tried}")

    def describe_batch(self, images: List[Image.Image], names: List[str] | None = None, types_list: List[str] | None = None) -> List[str]:
        # Prefer pipeline path if available (handles preprocessing internally)
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
                    # Sanitize special tokens and whitespace
                    txt = txt.replace("</s>", " ").replace("<s>", " ").replace("<pad>", " ")
                    txt = txt.replace("[PAD]", " ").replace("<unk>", " ")
                    txt = re.sub(r"\s+", " ", txt).strip()
                    txt = txt.strip(" .,;:-")
                    txt = polish_caption(txt)
                    caps.append(txt)
                return caps
            except Exception as e:
                if self.only_florence and self.debug:
                    print(f"[Florence pipeline error] {repr(e)}")
                if self.only_florence:
                    raise
                # fall through to native or generic pipeline below

        # Florence native path
        if self.florence_model is not None and self.florence_proc is not None:
            try:
                pure = getattr(self, "pure_mode", False)
                def build_instruction(name: str | None, types_str: str | None) -> str:
                    base = (
                        "Describe the Pokémon's appearance in detail: its color, pose, facial expression, unique body features (eyes, claws, wings, etc.), and any background elements or action."
                    )
                    if name and types_str:
                        return f"The image shows the Pokémon '{name}' (types: {types_str}). {base}"
                    if name:
                        return f"The image shows the Pokémon '{name}'. {base}"
                    return base

                captions: List[str] = [""] * len(images)

                import re

                def decode_and_clean(out_ids, used_prompt: str) -> List[str]:
                    if pure:
                        # Pure decode, minimal cleaning
                        txts = self.florence_proc.batch_decode(out_ids, skip_special_tokens=True)
                        return [t.strip() for t in txts]
                    txts = self.florence_proc.batch_decode(out_ids, skip_special_tokens=False)
                    cleaned: List[str] = []
                    for t in txts:
                        tt = str(t)
                        # Remove prompt echoes
                        if used_prompt:
                            tt = tt.replace(used_prompt, " ")
                        # Strip special tokens
                        tt = tt.replace("</s>", " ").replace("<s>", " ").replace("<pad>", " ")
                        tt = tt.replace("[PAD]", " ").replace("<unk>", " ")
                        # Collapse excessive whitespace and trim punctuation edges
                        tt = re.sub(r"\s+", " ", tt).strip()
                        tt = tt.strip(" .,;:-")
                        # Remove any residual instruction phrases
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

                # Prompting strategy
                need = [i for i, c in enumerate(captions) if not c]
                batch_imgs = [images[i] for i in need]
                if pure:
                    task_tok = f"<{getattr(self, 'pure_token', 'DETAILED_CAPTION')}>"
                    text_batch = [task_tok] * len(batch_imgs)
                else:
                    text_batch = []
                    for i in need:
                        nm = (names[i] if names and i < len(names) else None)
                        tp = (types_list[i] if types_list and i < len(types_list) else None)
                        text_batch.append(build_instruction(nm, tp))
                # Always build inputs after text_batch is ready
                inputs = self.florence_proc(text=text_batch, images=batch_imgs, return_tensors="pt")
                casted = {}
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        try:
                            import torch as _torch
                            if hasattr(v, 'dtype') and _torch.is_floating_point(v):
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
                partial = decode_and_clean(out, "")
                if self.debug:
                    for j, raw in enumerate(partial[:3]):
                        print(f"[Florence raw] {raw}")
                for idx, txt in zip(need, partial):
                    if txt and len(txt) >= 6:
                        captions[idx] = txt

                # per-image final retry with beams if still empty
                for i in range(len(captions)):
                    if captions[i] and len(captions[i]) >= 8:
                        continue
                    # Try per-image according to mode
                    if pure:
                        task_tok = f"<{getattr(self, 'pure_token', 'DETAILED_CAPTION')}>"
                        single_inputs = self.florence_proc(text=[task_tok], images=[images[i]], return_tensors="pt")
                    else:
                        nm = (names[i] if names and i < len(names) else None)
                        tp = (types_list[i] if types_list and i < len(types_list) else None)
                        inst = build_instruction(nm, tp)
                        single_inputs = self.florence_proc(text=[inst], images=[images[i]], return_tensors="pt")
                    casted_single = {}
                    for k, v in single_inputs.items():
                        if hasattr(v, 'to'):
                            try:
                                import torch as _torch
                                if hasattr(v, 'dtype') and _torch.is_floating_point(v):
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
                        out2 = self.florence_model.generate(**single_inputs, **alt_kwargs)
                    cap2 = decode_and_clean(out2, "")[0]
                    if self.debug:
                        print(f"[Florence retry beams] {cap2}")
                    captions[i] = cap2

                # ensure non-empty strings
                captions = [c if c else "" for c in captions]
                return captions
            except Exception as e:
                # If native path fails, fall back to pipeline if available
                if self.only_florence:
                    # Print detailed error in debug mode
                    if self.debug:
                        print(f"[Florence describe_batch error] {repr(e)}")
                    raise
                if self.debug:
                    print(f"[Florence describe_batch error] {repr(e)}")
                pass

        # Pipeline path
        assert self.pipe is not None
        try:
            outputs = self.pipe(images, batch_size=self.batch_size)
        except TypeError:
            outputs = self.pipe(images)

        captions: List[str] = []
        for out in outputs:
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
                captions.append(out[0]["generated_text"]) 
            elif isinstance(out, dict) and "generated_text" in out:
                captions.append(out["generated_text"])
            else:
                captions.append(str(out))
        return captions


def build_final_caption(folder: str, info: Dict[str, object], desc: str) -> str:
    color = str(info.get("color", "")) if isinstance(info, dict) else ""
    types = info.get("types", []) if isinstance(info, dict) else []
    if not isinstance(types, list):
        types = []
    type_str = " and ".join([str(t) for t in types if t]) + " Pokémon" if types else "Pokémon"

    # Clean and lowercase description sentence case
    desc = desc.strip()
    if desc and desc[0].isalpha():
        desc = desc[0].lower() + desc[1:]

    color_part = (color + " ") if color else ""
    if not desc:
        caption = f"{folder}, a {color_part}{type_str}."
    else:
        caption = f"{folder}, a {color_part}{type_str}, {desc}."

    # Fix double punctuation or trailing periods
    caption = caption.replace("..", ".")
    caption = caption.replace(" ,", ",")
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Path to dataset root with subfolders per Pokémon")
    parser.add_argument("--out_csv", default="dataset.csv", help="Output CSV path")
    parser.add_argument("--types_json", default="pokemon_types.json", help="Path to pokemon types JSON")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for captioning")
    parser.add_argument("--prefer_florence", action="store_true", help="Prefer microsoft/Florence-2-base when possible")
    parser.add_argument("--prefer_git", action="store_true", help="Prefer microsoft/git-large-coco before other fallbacks")
    parser.add_argument("--florence_model", default=None, help="Florence model id to use (e.g., microsoft/Florence-2-large-ft)")
    parser.add_argument("--only_florence", action="store_true", help="Only use Florence; do not fallback to other models")
    parser.add_argument("--debug_first_n", type=int, default=0, help="Process only first N images for debugging")
    parser.add_argument("--debug_print_raw", action="store_true", help="Print raw model outputs in console")
    parser.add_argument("--debug_write_raw", action="store_true", help="Append a raw column to CSV with unprocessed outputs")
    parser.add_argument("--florence_pure", action="store_true", help="Use Florence in pure mode: only task token, no prompt text, no post-processing")
    parser.add_argument("--florence_token", default="DETAILED_CAPTION", choices=["DETAILED_CAPTION", "CAPTION"], help="Task token to use in pure mode")
    parser.add_argument("--clip_refine", action="store_true", help="Use CLIP similarity to lightly refine species/action terms in captions")
    args = parser.parse_args()

    dataset_dir = args.dataset
    out_csv = args.out_csv
    types_json = args.types_json

    if Image is None:
        raise RuntimeError("Pillow is required. Please install with: pip install pillow")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Load or build types json
    types_map = load_or_build_types_json(dataset_dir, types_json)

    # Prepare captioner
    device = detect_device()
    prefer_florence = args.prefer_florence or (device == "cuda")
    captioner = Captioner(device=device, prefer_florence=prefer_florence, prefer_git=args.prefer_git, batch_size=max(8, min(args.batch_size, 16)), florence_model_id=args.florence_model, only_florence=args.only_florence, debug=args.debug_print_raw, clip_refine=bool(getattr(args, "clip_refine", False)))
    print(f"Using model: {captioner.model_name} on {device}")
    # Pass pure mode into captioner instance
    setattr(captioner, "pure_mode", bool(getattr(args, "florence_pure", False)))
    setattr(captioner, "pure_token", str(getattr(args, "florence_token", "DETAILED_CAPTION")))

    # Gather images
    items = list_images(dataset_dir)
    if args.debug_first_n and args.debug_first_n > 0:
        items = items[: args.debug_first_n]
    if not items:
        print("No images found.")
        return

    # CSV setup
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # Use UTF-8 with BOM so Excel detects encoding and shows accents correctly
    out_f = open(out_csv, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(out_f)
    header = ["image", "caption"]
    if args.debug_write_raw:
        header.append("raw")
    writer.writerow(header)

    # Batch processing
    errors = 0
    batch_imgs: List[Image.Image] = []
    batch_meta: List[Tuple[str, str]] = []  # (rel_path, folder)

    pbar = tqdm(total=len(items), desc="Captioning", unit="img")

    def flush_batch():
        nonlocal errors
        if not batch_imgs:
            return
        try:
            # Build names and types lists aligned with batch_meta
            names = [folder for (_, folder) in batch_meta]
            types_list = []
            for (_, folder) in batch_meta:
                info = types_map.get(folder, {})
                tps = info.get("types") or []
                types_list.append(", ".join(tps) if isinstance(tps, list) else str(tps))
            descs = captioner.describe_batch(batch_imgs, names=names, types_list=types_list)
        except Exception as e:
            # If batch fails, try one by one to salvage outputs
            if captioner.debug:
                print(f"[Batch error] {repr(e)}. Falling back to per-image generation…")
            descs = []
            for img in batch_imgs:
                try:
                    # Provide per-image name/types when retrying
                    nm = [batch_meta[len(descs)][1]] if len(descs) < len(batch_meta) else [None]
                    info = types_map.get(nm[0], {}) if nm[0] else {}
                    tps = info.get("types") or []
                    tp = [", ".join(tps) if isinstance(tps, list) else str(tps)] if nm[0] else None
                    d = captioner.describe_batch([img], names=nm, types_list=tp)[0]
                except Exception as ie:
                    d = ""
                    errors += 1
                    if captioner.debug:
                        print(f"[Per-image error] {repr(ie)}")
                descs.append(d)
        for (rel_path, folder), desc in zip(batch_meta, descs):
            info = types_map.get(folder, {})
            try:
                raw_desc = desc or ""
                # Optional CLIP refinement
                if getattr(captioner, "clip_refine", False):
                    try:
                        img_path = os.path.join(dataset_dir, rel_path)
                        with Image.open(img_path).convert("RGB") as _im:
                            raw_desc = captioner.clip_refine_desc(_im, raw_desc, folder)
                    except Exception as _e:
                        if captioner.debug:
                            print(f"[CLIP refine per-row failed] {repr(_e)}")
                raw_desc = clean_raw(raw_desc, folder)
                raw_desc = normalize_species(raw_desc, folder)
                # Build type_color string (e.g., "yellow Psychic Pokémon")
                color = (info.get("color") or "").strip()
                tps = info.get("types") or []
                type_str = (", ".join(tps) if isinstance(tps, list) else str(tps)).strip()
                type_color = f"{color} {type_str} Pokémon".strip()
                type_color = re.sub(r"\s+", " ", type_color)
                # Ensure lowercased first char of description
                body = raw_desc.strip()
                if body:
                    body = body[0].lower() + body[1:]
                final = generate_caption_from_template(folder, type_color, body)
                final = post_template_fix(final, folder)
            except Exception:
                final = f"{folder}, a Pokémon."
            row = [rel_path, final]
            if args.debug_write_raw:
                row.append(desc or "")
            writer.writerow(row)
        batch_imgs.clear()
        batch_meta.clear()

    bs = captioner.batch_size

    for rel_path, folder in items:
        abs_path = os.path.join(dataset_dir, rel_path)
        try:
            img = Image.open(abs_path).convert("RGB")
            img = resize_max_side(img, 768)
        except Exception:
            errors += 1
            pbar.update(1)
            continue
        batch_imgs.append(img)
        batch_meta.append((rel_path, folder))
        if len(batch_imgs) >= bs:
            flush_batch()
            pbar.update(bs)

    # flush remainder
    if batch_imgs:
        n = len(batch_imgs)
        flush_batch()
        pbar.update(n)

    out_f.close()
    pbar.close()

    print(f"Done. Wrote {out_csv}. Errors: {errors}")


if __name__ == "__main__":
    main()

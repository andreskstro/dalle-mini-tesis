import os
import argparse
import csv
from typing import List, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor


def get_device(explicit: str = None) -> torch.device:
    if explicit is not None:
        if explicit == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if explicit == "cpu":
            return torch.device("cpu")
        if explicit == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device: torch.device, cache_dir: str = None, blip_variant: str = "base"):
    clip_model_name = "openai/clip-vit-large-patch14"
    blip_model_name = (
        "Salesforce/blip-image-captioning-base" if blip_variant == "base" else "Salesforce/blip-image-captioning-large"
    )
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir, use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=cache_dir, use_safetensors=True)
    blip_processor = BlipProcessor.from_pretrained(blip_model_name, cache_dir=cache_dir)
    clip_model = clip_model.to(device)
    blip_model = blip_model.to(device)
    clip_model.eval()
    blip_model.eval()
    return clip_model, clip_processor, blip_model, blip_processor


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)


def build_text_embeds(clip_model: CLIPModel, clip_processor: CLIPProcessor, names: List[str], device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    prompts = []
    prompt_map = []
    for name in names:
        variants = [
            f"a photo of {name} Pokémon",
            f"Pokémon {name}",
            f"a {name} character",
        ]
        for p in variants:
            prompts.append(p)
            prompt_map.append(name)
    with torch.no_grad():
        text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_feats = clip_model.get_text_features(**text_inputs)
        text_feats = normalize(text_feats)
    name_to_vecs = {}
    for vec, name in zip(text_feats, prompt_map):
        name_to_vecs.setdefault(name, []).append(vec)
    name_embeds = []
    ordered_names = []
    for name in names:
        vecs = torch.stack(name_to_vecs[name], dim=0)
        mean_vec = normalize(vecs.mean(dim=0, keepdim=True))
        name_embeds.append(mean_vec)
        ordered_names.append(name)
    name_embeds = torch.cat(name_embeds, dim=0)
    return name_embeds, ordered_names


def detect_pokemon_in_image(img: Image.Image, clip_model: CLIPModel, clip_processor: CLIPProcessor, text_embeds: torch.Tensor, label_names: List[str], device: torch.device, top_k: int, threshold: float) -> List[Tuple[str, float]]:
    with torch.no_grad():
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        image_features = clip_model.get_image_features(**inputs)
        image_features = normalize(image_features)
        sims = (image_features @ text_embeds.t()).squeeze(0)
        values, idxs = torch.topk(sims, k=min(top_k, sims.shape[0]))
        results = []
        for score, idx in zip(values.tolist(), idxs.tolist()):
            if score >= threshold:
                results.append((label_names[idx], float(score)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def generate_blip_caption(img: Image.Image, blip_model: BlipForConditionalGeneration, blip_processor: BlipProcessor, device: torch.device, max_len: int = 30, num_beams: int = 3) -> str:
    with torch.no_grad():
        inputs = blip_processor(images=img, return_tensors="pt").to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = blip_model.generate(**inputs, max_length=max_len, num_beams=num_beams)
        else:
            out = blip_model.generate(**inputs, max_length=max_len, num_beams=num_beams)
        caption = blip_processor.batch_decode(out, skip_special_tokens=True)[0]
    return caption.strip()


def fuse_caption(pokemon_names: List[str], blip_caption: str) -> str:
    clause = blip_caption.strip()
    if clause.endswith("."):
        clause = clause[:-1]
    if not pokemon_names:
        return f"{clause}." if clause else ""
    if len(pokemon_names) == 1:
        subject = pokemon_names[0]
    elif len(pokemon_names) == 2:
        subject = f"{pokemon_names[0]} and {pokemon_names[1]}"
    else:
        subject = ", ".join(pokemon_names[:-1]) + f", and {pokemon_names[-1]}"
    if clause and clause[0].isupper():
        clause = clause[0].lower() + clause[1:]
    return f"{subject} {clause}." if clause else subject


def list_images(images_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = []
    for root, _, fnames in os.walk(images_dir):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                files.append(os.path.join(root, f))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--blip_model", type=str, choices=["base", "large"], default="base")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    images_dir = args.images_dir
    output_csv = args.output_csv or os.path.join(images_dir, "dataset.csv")

    pokemon_names = [
        "Abra","Aerodactyl","Alakazam","Arbok","Arcanine","Articuno","Beedrill","Bellsprout","Blastoise","Bulbasaur","Butterfree","Caterpie","Chansey","Charizard","Charmander","Charmeleon","Clefable","Clefairy","Cloyster","Cubone","Dewgong","Diglett","Ditto","Dodrio","Doduo","Dragonair","Dragonite","Dratini","Drowzee","Dugtrio","Eevee","Ekans","Electabuzz","Electrode","Exeggcute","Exeggutor","Farfetchd","Fearow","Flareon","Gastly","Gengar","Geodude","Gloom","Golbat","Goldeen","Golduck","Golem","Graveler","Grimer","Growlithe","Gyarados","Haunter","Hitmonchan","Hitmonlee","Horsea","Hypno","Ivysaur","Jigglypuff","Jolteon","Jynx","Kabuto","Kabutops","Kadabra","Kakuna","Kangaskhan","Kingler","Koffing","Krabby","Lapras","Lickitung","Machamp","Machoke","Machop","Magikarp","Magmar","Magnemite","Magneton","Mankey","Marowak","Meowth","Metapod","Mew","Mewtwo","Moltres","MrMime","Muk","Nidoking","Nidoqueen","Nidorina","Nidorino","Ninetales","Oddish","Omanyte","Omastar","Onix","Paras","Parasect","Persian","Pidgeot","Pidgeotto","Pidgey","Pikachu","Pinsir","Poliwag","Poliwhirl","Poliwrath","Ponyta","Porygon","Primeape","Psyduck","Raichu","Rapidash","Raticate","Rattata","Rhydon","Rhyhorn","Sandshrew","Sandslash","Scyther","Seadra","Seaking","Seel","Shellder","Slowbro","Slowpoke","Snorlax","Spearow","Squirtle","Starmie","Staryu","Tangela","Tauros","Tentacool","Tentacruel","Vaporeon","Venomoth","Venonat","Venusaur","Victreebel","Vileplume","Voltorb","Vulpix","Wartortle","Weedle","Weepinbell","Weezing","Wigglytuff","Zapdos","Zubat"
    ]

    device = get_device(args.device)
    clip_model, clip_processor, blip_model, blip_processor = load_models(device, args.cache_dir, blip_variant=args.blip_model)

    with torch.no_grad():
        text_embeds, label_names = build_text_embeds(clip_model, clip_processor, pokemon_names, device)

    image_paths = list_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    total = len(image_paths)
    written = 0

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        for i, img_path in enumerate(image_paths, 1):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            detected = detect_pokemon_in_image(img, clip_model, clip_processor, text_embeds, label_names, device, args.top_k, args.threshold)
            names_ranked = [n for n, s in detected]
            blip_caption = generate_blip_caption(img, blip_model, blip_processor, device)
            final_caption = fuse_caption(names_ranked, blip_caption)
            image_id = os.path.basename(img_path)
            writer.writerow([image_id, final_caption])
            written += 1
            if i % 25 == 0 or i == total:
                print(f"Processed {i}/{total} images")

    print(f"Saved {written} rows to {output_csv}")


if __name__ == "__main__":
    main()

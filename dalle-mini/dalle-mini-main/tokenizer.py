from dalle_mini.model.tokenizer import DalleBartTokenizer
tok = DalleBartTokenizer.from_pretrained("dalle-mini/dalle-mini")  # or local path to files
print(tok.vocab_size, tok.pad_token_id, tok.bos_token_id, tok.eos_token_id)

# 1) que el vocab coincida con el config
assert tok.vocab_size == 50265

# 2) que ningún prompt produzca IDs fuera de rango
enc = tok([" The image is of a Pokémon character with large, round eyes and a red bow on its head. "], padding="max_length", truncation=True, max_length=64, return_tensors="np")
assert int(enc["input_ids"].max()) < tok.vocab_size


# Muestra los IDs especiales
print("PAD ID:", tok.pad_token_id)
print("BOS ID:", tok.bos_token_id)
print("EOS ID:", tok.eos_token_id)
print("Decoder start ID:", tok.decoder_start_token_id)

# También puedes imprimir el mapeo completo
print(tok.special_tokens_map)
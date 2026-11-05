import json
p="tools/analyze/offline_topk_lm_head.jsonl"
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        rec=json.loads(line)
        if rec["neuron"]==42:
            print(rec["topk"]); break
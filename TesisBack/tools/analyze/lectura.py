import json
p="tools/analyze/offline_topk_lm_head.jsonl"
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        rec=json.loads(line)
        if rec["neuron"]==42:
            print(rec["topk"]); break

n=0
with open(p,"r",encoding="utf-8") as f:
    for _ in f: n+=1
print("neuronas:", n)
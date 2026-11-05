import argparse
import json
import heapq
from typing import Dict, List, Tuple, Iterable, Optional
from pathlib import Path


def iter_records(path: Path) -> Iterable[dict]:
    """Stream JSONL or JSON list records from disk."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for rec in data:
                    yield rec
            else:
                for rec in data.get("records", []):
                    yield rec


def top_images_per_neuron(
    records_path: Path,
    layer_name: str,
    neuron_indices: List[int],
    k: int = 5,
) -> Dict[int, List[Tuple[float, str]]]:
    """Return top-k images per neuron index based on stored per-neuron influences.

    The file must contain entries like:
      {"image_id": "31778", "layer_influences": {"lm_head.kernel": [..]}}
    """
    heaps: Dict[int, List[Tuple[float, str]]] = {n: [] for n in neuron_indices}

    for rec in iter_records(records_path):
        image_id = rec.get("image_id") or rec.get("filename") or rec.get("id")
        if not image_id:
            continue
        li = rec.get("layer_influences", {})
        vec = li.get(layer_name)
        if not isinstance(vec, list):
            continue
        vec_len = len(vec)
        for n in neuron_indices:
            if 0 <= n < vec_len:
                score = float(vec[n])
                h = heaps[n]
                if len(h) < k:
                    heapq.heappush(h, (score, image_id))
                else:
                    if score > h[0][0]:
                        heapq.heapreplace(h, (score, image_id))

    return {n: sorted(h, key=lambda x: x[0], reverse=True) for n, h in heaps.items()}


def top_images_aggregate(
    records_path: Path,
    layer_name: str,
    neuron_indices: List[int],
    neuron_weights: Optional[List[float]] = None,
    k: int = 5,
) -> List[Tuple[float, str]]:
    """Return top-k images aggregating across a set of neurons with optional weights.

    Aggregate score per image = sum_j (alpha_j * influence_j(image)), where influence_j(image)
    is the per-neuron influence stored for layer_name at index j, and alpha_j are the weights.
    """
    if neuron_weights is None:
        neuron_weights = [1.0] * len(neuron_indices)
    assert len(neuron_weights) == len(neuron_indices), "neuron_weights must match neuron_indices length"

    # Min-heap for global top-k
    heap: List[Tuple[float, str]] = []

    for rec in iter_records(records_path):
        image_id = rec.get("image_id") or rec.get("filename") or rec.get("id")
        if not image_id:
            continue
        li = rec.get("layer_influences", {})
        vec = li.get(layer_name)
        if not isinstance(vec, list):
            continue
        vec_len = len(vec)
        agg = 0.0
        for idx, alpha in zip(neuron_indices, neuron_weights):
            if 0 <= idx < vec_len:
                agg += float(vec[idx]) * float(alpha)
        # Update heap
        if len(heap) < k:
            heapq.heappush(heap, (agg, image_id))
        else:
            if agg > heap[0][0]:
                heapq.heapreplace(heap, (agg, image_id))

    return sorted(heap, key=lambda x: x[0], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Query top images per neuron from stored influence history")
    parser.add_argument("--influences", type=str, required=True, help="Path to influence JSONL/JSON produced by compute_influence.py")
    parser.add_argument("--layer", type=str, default="lm_head.kernel", help="Layer name to query (e.g., lm_head.kernel)")
    parser.add_argument("--indices", type=int, nargs="+", required=True, help="Neuron indices to query (from EB top-k)")
    parser.add_argument("--weights", type=float, nargs="*", help="Optional weights (EB values) aligned with indices for aggregate query")
    parser.add_argument("--k", type=int, default=5, help="Top-k images to return")
    parser.add_argument("--aggregate", action="store_true", help="If set, also compute aggregate top-k over all indices (weighted if --weights provided)")

    args = parser.parse_args()
    records_path = Path(args.influences)

    # Top-k per neuron
    per_neuron = top_images_per_neuron(records_path, args.layer, args.indices, args.k)
    print("Per-neuron top-{} results:".format(args.k))
    for n in args.indices:
        print(f"Neuron {n}:")
        for score, image_id in per_neuron.get(n, []):
            print(f"  {image_id}\t{score:.6f}")

    # Aggregate
    if args.aggregate:
        agg = top_images_aggregate(records_path, args.layer, args.indices, args.weights, args.k)
        print("\nAggregate top-{} results:".format(args.k))
        for score, image_id in agg:
            print(f"  {image_id}\t{score:.6f}")


if __name__ == "__main__":
    main()

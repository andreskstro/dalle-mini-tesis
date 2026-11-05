export type NeuronTopKItem = {
  filename: string;
  score: number;
  caption?: string;
};

export type NeuronTopKResponse = {
  neuron: number;
  topk: NeuronTopKItem[];
};

export async function fetchNeuronTopK(neuronId: number, k = 5): Promise<NeuronTopKResponse> {
  const r = await fetch(`/api/neuron/${neuronId}/topk?k=${k}`);
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = (j && j.detail) || detail; } catch {}
    throw new Error(`TopK neurona ${neuronId}: ${r.status} ${detail}`);
  }
  return r.json();
}

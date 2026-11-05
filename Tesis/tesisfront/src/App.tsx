import React, { useEffect, useState } from 'react';
import PromptForm from './components/PromptForm';
import { fetchNeuronTopK } from './api/topk';
import DatasetStrip, { DatasetItem } from './components/DatasetStrip';

const API_BASE = (import.meta as any)?.env?.VITE_BACKEND_BASE || 'http://localhost:8000';

// helper seguro (evita `any` en el catch)
function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  return 'Error generando imágenes';
}

function NeuronTopK({ neuronId, k = 5 }: { neuronId: number; k?: number }) {
  const [items, setItems] = useState<{ filename: string; score: number; caption?: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // simple effect per neuronId change
  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);
    fetchNeuronTopK(neuronId, k)
      .then((res) => {
        if (!cancelled) setItems(res.topk || []);
      })
      .catch((e) => !cancelled && setErr(e?.message || 'Error cargando top‑k'))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [neuronId, k]);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-medium">Neurona {neuronId}</h3>
        {loading && <span className="text-xs text-white/60">cargando…</span>}
      </div>
      {err && <div className="text-xs text-red-300">{err}</div>}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3">
        {items.map((it) => (
          <div key={it.filename} className="rounded-lg overflow-hidden border border-white/10 bg-black/20">
            <img src={`${API_BASE}/image/${it.filename}`} alt={it.caption || it.filename} className="w-full h-32 object-cover" />
            <div className="p-2 text-xs">
              <div className="text-white/80">score: {it.score.toFixed(2)}</div>
              <div className="text-white/60 line-clamp-2">{it.caption}</div>
            </div>
          </div>
        ))}
        {items.length === 0 && !loading && <div className="text-xs text-white/60">Sin resultados</div>}
      </div>
    </div>
  );
}

// Fondo que usábamos (gradientes + glass) – dentro del mismo archivo
function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10">
      <div className="absolute inset-0 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950" />
      <div className="absolute -top-28 -left-28 h-[42rem] w-[42rem] rounded-full bg-blue-500/25 blur-3xl" />
      <div className="absolute -bottom-28 -right-28 h-[42rem] w-[42rem] rounded-full bg-pink-500/25 blur-3xl" />
    </div>
  );
}

export default function App() {
  const [images, setImages] = useState<string[]>([]);
  const [imageIds, setImageIds] = useState<string[]>([]);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ebIndices, setEbIndices] = useState<number[] | null>(null);
  const [ebValues, setEbValues] = useState<number[] | null>(null);
  const [datasetItems, setDatasetItems] = useState<DatasetItem[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetItem | null>(null);
  // const [similarities, setSimilarities] = useState<(number | null)[]>([]);
  const [generationStart, setGenerationStart] = useState<number | null>(null);
  const [generationTime, setGenerationTime] = useState<number | null>(null);

  // Update timer every second during generation
  useEffect(() => {
    if (loading && generationStart !== null) {
      const interval = setInterval(() => {
        const elapsed = (Date.now() - generationStart) / 1000;
        setGenerationTime(elapsed);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [loading, generationStart]);

  // load first 5 dataset images
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/dataset/first?n=7');
        if (!res.ok) throw new Error(`Error ${res.status}`);
        const data = await res.json();
        const items = (data?.items ?? []).map((it: any) => ({
          id: String(it.filename),
          src: `${API_BASE}${it.image_url}`,
          label: typeof it.caption === 'string' && it.caption.trim().length > 0 ? String(it.caption) : String(it.filename),
        })) as DatasetItem[];
        if (!cancelled) {
          setDatasetItems(items);
          if (items.length > 0) setSelectedDataset(items[0]);   
        }
      } catch (e) {
        // ignore silently
      }
    })();
    return () => { cancelled = true; };
  }, []);

  async function handleGenerate(prompt: string, numImages: number) {
    setError(null);
    setLoading(true);
    setGenerationStart(Date.now());
    setGenerationTime(null);
    setImages([]);
    setImageIds([]);
    setBatchId(null);
    setEbIndices(null);
    setEbValues(null);
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, n_predictions: numImages }),
      });
      if (!res.ok) {
        try {
          const j = await res.json();
          const detail = typeof j?.detail === 'string' ? j.detail : JSON.stringify(j);
          throw new Error(`Error ${res.status}: ${detail || res.statusText}`);
        } catch (_) {
          const text = await res.text().catch(() => '');
          throw new Error(`Error ${res.status}: ${text || res.statusText}`);
        }
      }
      const data = await res.json();
      const imgs = Array.isArray((data as any)?.images) ? (data as any).images : [];
      const ids = Array.isArray((data as any)?.image_ids) ? (data as any).image_ids : [];
      const bid = typeof (data as any)?.batch_id === 'string' ? (data as any).batch_id : null;
      setImages(imgs);
      setImageIds(ids);
      setBatchId(bid);
      // setSimilarities(new Array(imgs.length).fill(null));
    } catch (err: unknown) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  /* async function handleCompare(index: number) {
    if (!selectedDataset) {
      setError('Selecciona una imagen del dataset primero');
      return;
    }
    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_filename: selectedDataset.id,
          generated_image_data_url: images[index],
          method: 'clip',
        }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || `Error ${res.status}`);
      }
      const data = await res.json();
      const sim = typeof data?.sim_clip_01 === 'number' ? data.sim_clip_01 : null;
      setSimilarities((prev) => {
        const next = [...prev];
        next[index] = sim;
        return next;
      });
    } catch (e: any) {
      setError(getErrorMessage(e));
    }
  } */

  async function handleAnalyze(index: number) {
    if (!batchId || !imageIds[index]) return;
    setError(null);
    // Don't set loading to true to avoid timer reset during analysis
    // setLoading(true);
    setEbIndices(null);
    setEbValues(null);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch_id: batchId, image_id: imageIds[index], k: 6 }),
      });
      if (!res.ok) {
        try {
          const j = await res.json();
          const detail = typeof j?.detail === 'string' ? j.detail : JSON.stringify(j);
          throw new Error(`Error ${res.status}: ${detail || res.statusText}`);
        } catch (_) {
          const text = await res.text().catch(() => '');
          throw new Error(`Error ${res.status}: ${text || res.statusText}`);
        }
      }
      const data = await res.json();
      setEbIndices(Array.isArray((data as any)?.eb_top_indices) ? (data as any).eb_top_indices : null);
      setEbValues(Array.isArray((data as any)?.eb_top_values) ? (data as any).eb_top_values : null);
    } catch (err: unknown) {
      setError(getErrorMessage(err));
    }
  }

  return (
    <div className="min-h-screen text-white">
      <Background />

      <header className="border-b border-white/10 bg-black/20 backdrop-blur">
        <div className="max-w-5xl mx-auto p-6">
          <h1 className="text-2xl font-semibold">Generador — DALL·E mini</h1>
        </div>
      </header>

      <main className="max-w-5xl mx-auto p-6 space-y-6">
        {/* Dataset selector (first 5) */}
        <section className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-4">
          <h2 className="text-sm font-medium text-white/80 mb-2">Selecciona imagen del dataset</h2>
          <DatasetStrip
            items={datasetItems}
            selectedId={selectedDataset?.id ?? null}
            onSelect={(it) => setSelectedDataset(it)}
          />
        </section>
        <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-5">
          <PromptForm
            onGenerate={handleGenerate}
            loading={loading}
            prefillPrompt={selectedDataset?.label}
          />
        </div>

        {error && (
          <div className="rounded-2xl border border-red-400/20 bg-red-500/10 p-3 text-red-200">
          </div>
        )}

        {loading && (
          <div className="text-sm text-white/70">
            Generando…
          </div>
        )}
        
        

        {/* Galería con handler de análisis por imagen */}
        <section className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {images.map((src, i) => (
            <figure
              key={i}
              className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 backdrop-blur group"
            >
              <img src={src} alt={`gen-${i}`} className="w-full h-64 object-cover" />
              {/* similarity badge */}
              {/* {typeof similarities[i] === 'number' && (
                <div className="absolute top-2 left-2 rounded-md bg-black/60 px-2 py-1 text-xs">
                  sim: {(similarities[i]! as number).toFixed(2)}
                </div>
              )} */}
              <div className="absolute inset-0 flex items-start justify-end p-3 gap-2 opacity-0 group-hover:opacity-100 transition">
                <button
                  onClick={() => handleAnalyze(i)}
                  className="rounded-lg border border-white/20 bg-white/80 px-3 py-1 text-xs text-slate-900"
                >
                  Analizar EB
                </button>
                {/* <button
                  onClick={() => handleCompare(i)}
                  className="rounded-lg border border-white/20 bg-white/80 px-3 py-1 text-xs text-slate-900"
                >
                  Comparar imagen
                </button> */}
              </div>
            </figure>
          ))}
        </section>

        {/* Resultado EB */}
        {ebIndices && ebValues && (
          <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-4 text-sm">
            <div className="font-medium mb-2">EB resultado (top-k en hidden antes de lm_head)</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="text-white/70">Indices</div>
                <pre className="mt-1 whitespace-pre-wrap break-all">{JSON.stringify(ebIndices)}</pre>
              </div>
              <div>
                <div className="text-white/70">Valores</div>
                <pre className="mt-1 whitespace-pre-wrap break-all">{JSON.stringify(ebValues)}</pre>
              </div>
            </div>
          </div>
        )}

        {/* Top-5 por neurona (historial de influencia VJP) */}
        {ebIndices && (
          <section className="space-y-6">
            <h2 className="text-lg font-semibold">Top‑5 por neurona (historial de influencia)</h2>
            <div className="space-y-8">
              {ebIndices.slice(0, 6).map((nid) => (
                <NeuronTopK key={nid} neuronId={nid} k={5} />
              ))}
            </div>
          </section>
        )}

        {!loading && images.length === 0 && !error && (
          <p className="text-sm text-white/60">Escribe un prompt y pulsa Generar 🙂</p>
        )}
      </main>
    </div>
  );
}
import React, { useEffect, useState } from 'react';

type Props = {
  onGenerate: (prompt: string, numImages: number) => Promise<void> | void;
  loading?: boolean;
  prefillPrompt?: string;
};

export default function PromptForm({ onGenerate, loading, prefillPrompt }: Props) {
  const [prompt, setPrompt] = useState('');

  // prefill when provided by parent (e.g., from dataset selection)
  useEffect(() => {
    if (typeof prefillPrompt === 'string' && prefillPrompt.length > 0) {
      setPrompt(prefillPrompt);
    }
  }, [prefillPrompt]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    await onGenerate(prompt, 9);
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="space-y-2">
        <label className="block text-sm font-medium text-white/80">Prompt</label>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Escribe tu prompt…"
          className="w-full rounded-xl border border-white/10 bg-white/10 text-white placeholder-white/50 px-4 py-3 outline-none focus:ring-2 focus:ring-white/30"
          required
        />
      </div>

      <div className="flex items-center gap-3">
        <label className="text-sm text-white/70">Cantidad</label>
        <input
          type="number"
          min={9}
          max={9}
          value={9}
          readOnly
          disabled
          className="w-24 rounded-xl border border-white/10 bg-white/10 text-white px-3 py-2 opacity-60 cursor-not-allowed"
        />
        <div className="flex-1" />
        <button
          type="submit"
          disabled={loading}
          className="rounded-xl bg-white text-slate-900 px-5 py-3 font-medium disabled:opacity-50"
        >
          {loading ? 'Generando…' : 'Generar'}
        </button>
      </div>
    </form>
  );
}
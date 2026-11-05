export default function SkeletonCard() {
  return (
    <div className="animate-pulse overflow-hidden rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60">
      <div className="aspect-square w-full bg-zinc-200/60 dark:bg-zinc-800/60" />
      <div className="p-3">
        <div className="h-3 w-3/4 rounded bg-zinc-200/70 dark:bg-zinc-800/70" />
      </div>
    </div>
  );
}

import ThemeToggle from "./ThemeToggle";

export default function Header() {
  return (
    <header className="sticky top-0 z-40 w-full bg-white/70 backdrop-blur dark:bg-zinc-950/60 border-b border-zinc-200 dark:border-zinc-800">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 via-violet-500 to-fuchsia-500 shadow-lg" />
          <div>
            <p className="text-lg font-semibold tracking-tight">
              DalleMini Studio
            </p>
            <p className="text-xs text-zinc-500">
              Genera imágenes con tu prompt
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

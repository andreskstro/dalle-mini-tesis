export default function EmptyState() {
  return (
    <div className="col-span-full flex flex-col items-center justify-center rounded-2xl border border-dashed border-zinc-300 dark:border-zinc-800 py-14 text-center">
      <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 opacity-80 mb-4" />
      <h3 className="text-lg font-semibold mb-1">Aún no hay imágenes</h3>
      <p className="text-sm text-zinc-500 max-w-sm">
        Escribe un prompt a la izquierda y presiona{" "}
        <span className="font-medium text-zinc-700 dark:text-zinc-300">
          Generar
        </span>
        .
      </p>
    </div>
  );
}

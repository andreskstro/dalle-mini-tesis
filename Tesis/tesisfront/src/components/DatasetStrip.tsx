import React from "react";

export type DatasetItem = {
  id: string;
  src: string;
  label?: string;
};

interface Props {
  items: DatasetItem[];
  selectedId?: string | null;
  onSelect?: (item: DatasetItem) => void;
}

export default function DatasetStrip({ items, selectedId, onSelect }: Props) {
  if (!items || items.length === 0) {
    return (
      <div className="text-center text-sm text-zinc-500 dark:text-zinc-400 py-4">
        No hay imágenes de referencia aún.
      </div>
    );
  }

  return (
    <div className="flex gap-3 overflow-x-auto py-4">
      {items.map((item) => {
        const isSel = selectedId === item.id;
        return (
          <button
            key={item.id}
            onClick={() => onSelect && onSelect(item)}
            className={`flex-shrink-0 w-32 rounded-lg overflow-hidden border ${
              isSel ? 'border-blue-500 ring-2 ring-blue-500/40' : 'border-zinc-200 dark:border-zinc-800'
            } focus:outline-none`}
            title={item.label ?? item.id}
          >
            <img
              src={item.src}
              alt={item.label ?? 'dataset'}
              className="w-full h-24 object-cover"
            />
            {item.label && (
              <p className="text-xs text-center text-zinc-600 dark:text-zinc-300 py-1">
                {item.label}
              </p>
            )}
          </button>
        );
      })}
    </div>
  );
}

type Props = {
  src: string;
  prompt: string;
};

export default function ImageCard({ src, prompt }: Props) {
  return (
    <figure className="group overflow-hidden rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-900/60 shadow-sm">
      <img
        src={src}
        alt={prompt}
        className="aspect-square w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
      />
      <figcaption className="p-3 text-xs text-zinc-600 dark:text-zinc-300 line-clamp-2">
        {prompt}
      </figcaption>
    </figure>
  );
}

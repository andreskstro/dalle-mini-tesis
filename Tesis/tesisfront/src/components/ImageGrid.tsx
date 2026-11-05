import ImageCard from "./ImageCard";
import SkeletonCard from "./SkeletonCard";
import EmptyState from "./EmptyState";

type Item = { id: string; src: string; prompt: string };

type Props = {
  items: Item[];
  loading?: boolean;
};

export default function ImageGrid({ items, loading }: Props) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (!items.length) {
    return (
      <div className="grid grid-cols-1">
        <EmptyState />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
      {items.map((it) => (
        <ImageCard key={it.id} src={it.src} prompt={it.prompt} />
      ))}
    </div>
  );
}

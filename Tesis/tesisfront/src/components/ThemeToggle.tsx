import { useEffect, useState } from "react";

function getInitialTheme(): "dark" | "light" {
  if (typeof window === "undefined") return "light";
  const saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light") return saved as "dark" | "light";
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">(getInitialTheme);

  // aplica/remueve en <html>
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    root.setAttribute("data-theme", theme); // útil si luego quieres selectores por atributo
    localStorage.setItem("theme", theme);
  }, [theme]);

  // opcional: escucha cambios del sistema y respétalos si no hay override manual
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      const saved = localStorage.getItem("theme");
      if (!saved) setTheme(mq.matches ? "dark" : "light");
    };
    mq.addEventListener?.("change", handler);
    return () => mq.removeEventListener?.("change", handler);
  }, []);

  const isDark = theme === "dark";

  return (
    <button
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label="Cambiar tema"
      className="inline-flex items-center gap-2 rounded-full border border-zinc-200 dark:border-zinc-800 px-3 py-1 text-xs text-zinc-700 dark:text-zinc-200 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition"
    >
      <span className="h-2.5 w-2.5 rounded-full bg-yellow-400 dark:bg-zinc-200" />
      {isDark ? "Claro" : "Oscuro"}
    </button>
  );
}

import type { Config } from "tailwindcss";

export default {
  darkMode: "class", // ← imprescindible
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;

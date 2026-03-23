import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FIX: file was named vite_config.js — Vite requires the name vite.config.js
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { mockJarvisServer } from './src/mock/mockServer';

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    tailwindcss(),
    // The mock engine is opt-in: `npm run dev:mock` (vite --mode mock). Default
    // dev/build connects to the real Python bridge (ui/bridge.py).
    ...(mode === 'mock' ? [mockJarvisServer()] : []),
  ],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ['VITE_', 'TAURI_'],
}));

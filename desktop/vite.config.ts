import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { mockJarvisServer } from './src/mock/mockServer';

export default defineConfig({
  plugins: [react(), tailwindcss(), mockJarvisServer()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ['VITE_', 'TAURI_'],
});

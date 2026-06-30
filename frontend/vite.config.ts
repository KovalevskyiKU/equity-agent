import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: proxy /api (and the /ws WebSocket) to the FastAPI backend (eqa serve, port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})

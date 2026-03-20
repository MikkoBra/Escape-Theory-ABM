import { defineConfig } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    host: '0.0.0.0',       // expose outside container
    port: 5173,
    watch: {
      usePolling: true,    // required for Docker volume mounts on Mac/Windows
    },
    proxy: {
      '/api': {
        target: 'http://api:5000',   // 'api' = service name in docker-compose
        changeOrigin: true,
      }
    }
  }
})

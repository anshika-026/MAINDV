import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Fixed, non-default port — deco-vision (a separate, unrelated project)
    // runs its own dev server on 5173 on this same machine.
    port: 5180,
    strictPort: true,
  },
})

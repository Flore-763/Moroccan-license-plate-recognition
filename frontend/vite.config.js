import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' : le site fonctionne sur n'importe quel chemin (GitHub Pages /repo/, Netlify, etc.)
export default defineConfig({
  base: './',
  plugins: [react()],
})

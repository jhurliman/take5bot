import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  // Base must match the repository name for GitHub Pages when using <user>.github.io/<repo>
  // If deploying to a custom domain or user/organization page, adjust/remove accordingly.
  base: '/take5bot/',
  plugins: [
    react(),
    tailwindcss(),
  ],
})

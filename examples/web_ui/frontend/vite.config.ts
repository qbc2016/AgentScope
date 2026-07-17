import path from 'path';

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import svgr from 'vite-plugin-svgr';

export default defineConfig({
	plugins: [react(), tailwindcss(), svgr()],
	base: './',
	server: {
		port: 5173,
		strictPort: true,
		proxy: {
			'/api': 'http://localhost:3000',
		},
	},
	build: {
		outDir: '../electron-dist/frontend-dist',
		emptyOutDir: true,
	},
	resolve: {
		alias: {
			'@': path.resolve(__dirname, './src'),
			'next/navigation': path.resolve(__dirname, './src/lib/next-navigation-shim.ts'),
		},
	},
	optimizeDeps: {
		include: ['mime-types'],
	},
});

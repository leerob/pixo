/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			fontFamily: {
				mono: ['Berkeley Mono', 'SF Mono', 'Fira Code', 'monospace']
			},
			colors: {
				surface: {
					0: '#0a0a0a',
					1: '#111111',
					2: '#1a1a1a',
					3: '#222222'
				},
				border: {
					DEFAULT: '#2a2a2a',
					hover: '#333333'
				},
				terminal: {
					green: '#22c55e',
					red: '#ef4444'
				}
			}
		}
	},
	plugins: []
};

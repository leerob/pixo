import { test as base, expect } from 'playwright/test';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Path to test fixtures
 */
export const FIXTURES = {
	PNG: join(__dirname, '..', '..', 'tests', 'fixtures', 'playground.png'),
	JPEG: join(__dirname, '..', '..', 'tests', 'fixtures', 'multi-agent.jpg'),
} as const;

/**
 * Extended test with common utilities
 */
export const test = base.extend<{
	/** Wait for WASM to be fully initialized */
	waitForWasm: () => Promise<void>;
	/** Upload file(s) and wait for processing to complete */
	uploadAndWaitForCompression: (filePaths: string | string[]) => Promise<void>;
}>({
	waitForWasm: async ({ page }, use) => {
		const waitForWasm = async () => {
			// Wait for the footer to indicate WASM is ready (no loading indicator)
			await expect(page.getByTestId('controls-footer')).toHaveAttribute('data-wasm-ready', 'true', {
				timeout: 30000,
			});
		};
		await use(waitForWasm);
	},

	uploadAndWaitForCompression: async ({ page }, use) => {
		const uploadAndWait = async (filePaths: string | string[]) => {
			const paths = Array.isArray(filePaths) ? filePaths : [filePaths];
			
			// Upload files
			const fileInput = page.getByTestId('file-input');
			await fileInput.setInputFiles(paths);
			
			// Wait for compression to complete - download button appears when at least one job is done
			await expect(page.getByTestId('download-button')).toBeVisible({
				timeout: 60000, // WASM compression can be slow
			});
		};
		await use(uploadAndWait);
	},
});

export { expect };

/**
 * Helper to parse bytes from formatted string (e.g., "123 KB" -> 123000)
 */
export function parseFormattedBytes(formatted: string): number {
	const match = formatted.match(/^([\d.]+)\s*(B|KB|MB|GB)$/);
	if (!match) return 0;
	
	const value = parseFloat(match[1]);
	const unit = match[2];
	const multipliers: Record<string, number> = {
		'B': 1,
		'KB': 1024,
		'MB': 1024 * 1024,
		'GB': 1024 * 1024 * 1024,
	};
	
	return Math.round(value * (multipliers[unit] || 1));
}

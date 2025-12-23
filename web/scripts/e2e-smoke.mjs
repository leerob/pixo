import { chromium } from 'playwright';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const appDir = join(__dirname, '..');
const fixturePath = join(__dirname, '..', '..', 'tests', 'fixtures', 'playground.png');
const baseUrl = process.env.BASE_URL || 'http://localhost:4173/';

async function run() {
	const browser = await chromium.launch({ headless: true });
	const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

	await page.goto(baseUrl, { waitUntil: 'networkidle' });
	await page.setInputFiles('input[type=file]', fixturePath);
	await page.waitForSelector('text=playground.png', { timeout: 10000 });
	// Compression happens automatically when files are added
	// Wait for the Download button to appear (indicates compression is complete)
	await page.waitForSelector('text=Download', { timeout: 30000 });
	await page.screenshot({ path: 'playwright-screenshot.png', fullPage: true });

	await browser.close();
	console.log('[e2e] Smoke test passed');
}

run().catch((err) => {
	console.error('[e2e] Failed:', err.message);
	process.exit(1);
});

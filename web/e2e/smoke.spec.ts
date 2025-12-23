import { test, expect, FIXTURES } from './fixtures';

test.describe('Smoke Tests', () => {
	test.describe('Application Loading', () => {
		test('should load the application and show drop zone', async ({ page }) => {
			await page.goto('/');
			
			// Verify app container is present
			await expect(page.getByTestId('app-container')).toBeVisible();
			
			// Verify drop zone is shown initially
			await expect(page.getByTestId('drop-zone')).toBeVisible();
			await expect(page.getByTestId('select-files-button')).toBeVisible();
			
			// Verify main content shows drop view mode
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
		});

		test('should initialize WASM successfully', async ({ page, waitForWasm }) => {
			await page.goto('/');
			
			// Initially might show loading state
			const wasmLoading = page.getByTestId('wasm-loading');
			const controlsFooter = page.getByTestId('controls-footer');
			
			// Wait for WASM to be ready
			await waitForWasm();
			
			// Loading indicator should not be visible when WASM is ready
			await expect(wasmLoading).not.toBeVisible();
			await expect(controlsFooter).toHaveAttribute('data-wasm-ready', 'true');
		});
	});

	test.describe('PNG Compression', () => {
		test('should upload and compress a PNG file', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			// Upload PNG file
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Should switch to single view mode
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'single');
			await expect(page.getByTestId('single-view')).toBeVisible();
			
			// Verify image info is displayed
			await expect(page.getByTestId('image-name')).toContainText('playground.png');
			await expect(page.getByTestId('image-dimensions')).toBeVisible();
			
			// Verify compressed image overlay is shown
			await expect(page.getByTestId('compressed-image-overlay')).toBeVisible();
			await expect(page.getByTestId('comparison-slider')).toBeVisible();
		});

		test('should show compression stats for PNG', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Check stats are visible
			await expect(page.getByTestId('total-original-size')).toBeVisible();
			await expect(page.getByTestId('total-compressed-size')).toBeVisible();
			await expect(page.getByTestId('total-savings')).toBeVisible();
			
			// Savings should be a percentage
			const savingsText = await page.getByTestId('total-savings').textContent();
			expect(savingsText).toMatch(/^[+-][\d.]+%$/);
		});

		test('should allow changing PNG compression settings', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Verify PNG controls are visible
			await expect(page.getByTestId('format-select')).toHaveValue('png');
			await expect(page.getByTestId('compression-level-slider')).toBeVisible();
			await expect(page.getByTestId('filter-select')).toBeVisible();
			
			// Get initial compressed size
			const initialSize = await page.getByTestId('total-compressed-size').textContent();
			
			// Change compression level to maximum
			await page.getByTestId('compression-level-slider').fill('9');
			await page.getByTestId('compression-level-slider').dispatchEvent('change');
			
			// Wait for recompression
			await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 60000 });
			
			// Size should have potentially changed (or stayed same, but UI should still work)
			await expect(page.getByTestId('total-compressed-size')).toBeVisible();
		});
	});

	test.describe('JPEG Compression', () => {
		test('should upload and compress a JPEG file', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			// Upload JPEG file
			await uploadAndWaitForCompression(FIXTURES.JPEG);
			
			// Should switch to single view mode
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'single');
			await expect(page.getByTestId('single-view')).toBeVisible();
			
			// Verify image info is displayed
			await expect(page.getByTestId('image-name')).toContainText('multi-agent.jpg');
			
			// Should auto-select JPEG format for JPEG input
			await expect(page.getByTestId('format-select')).toHaveValue('jpeg');
		});

		test('should show JPEG quality controls for JPEG format', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.JPEG);
			
			// Verify JPEG controls are visible
			await expect(page.getByTestId('quality-slider')).toBeVisible();
			await expect(page.getByTestId('quality-value')).toBeVisible();
			
			// PNG-specific controls should not be visible
			await expect(page.getByTestId('compression-level-slider')).not.toBeVisible();
			await expect(page.getByTestId('filter-select')).not.toBeVisible();
		});
	});

	test.describe('Format Switching', () => {
		test('should switch format and recompress', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			// Upload PNG
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Should be PNG format initially
			await expect(page.getByTestId('format-select')).toHaveValue('png');
			
			// Switch to JPEG
			await page.getByTestId('format-select').selectOption('jpeg');
			
			// Wait for recompression
			await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 60000 });
			
			// JPEG controls should now be visible
			await expect(page.getByTestId('quality-slider')).toBeVisible();
		});
	});

	test.describe('Download', () => {
		test('should enable download button after compression', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Download button should be enabled
			const downloadButton = page.getByTestId('download-button');
			await expect(downloadButton).toBeVisible();
			await expect(downloadButton).toBeEnabled();
			await expect(downloadButton).toHaveText('Download');
		});

		test('should trigger download when clicking download button', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Set up download listener
			const downloadPromise = page.waitForEvent('download', { timeout: 10000 });
			
			// Click download
			await page.getByTestId('download-button').click();
			
			// Verify download was triggered
			const download = await downloadPromise;
			expect(download.suggestedFilename()).toContain('playground');
			expect(download.suggestedFilename()).toContain('-compressed');
		});
	});

	test.describe('Image Comparison Slider', () => {
		test('should have working comparison slider', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Slider should be visible
			const slider = page.getByTestId('comparison-range-slider');
			await expect(slider).toBeVisible();
			
			// Get initial value
			const initialValue = await slider.inputValue();
			expect(parseInt(initialValue)).toBe(50); // Default is 50%
			
			// Change slider value
			await slider.fill('75');
			await slider.dispatchEvent('input');
			
			// Verify value changed
			await expect(slider).toHaveValue('75');
		});
	});

	test.describe('Remove/Close', () => {
		test('should remove image and return to drop zone', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			
			await uploadAndWaitForCompression(FIXTURES.PNG);
			
			// Click close button
			await page.getByTestId('close-button').click();
			
			// Should return to drop zone
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
			await expect(page.getByTestId('drop-zone')).toBeVisible();
		});
	});
});

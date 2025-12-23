import { test, expect, FIXTURES } from './fixtures';

test.describe('Keyboard Navigation', () => {
	test('should open file picker with Cmd/Ctrl+O', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Set up file chooser listener
		const fileChooserPromise = page.waitForEvent('filechooser', { timeout: 5000 });
		
		// Press Cmd+O (or Ctrl+O on non-Mac)
		await page.keyboard.press('Control+o');
		
		// File chooser should open
		const fileChooser = await fileChooserPromise;
		expect(fileChooser).toBeTruthy();
		
		// Cancel the file chooser
		await fileChooser.setFiles([]);
	});

	test('should close single view with Escape when only one file', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
		await page.goto('/');
		await waitForWasm();
		
		await uploadAndWaitForCompression(FIXTURES.PNG);
		
		// Should be in single view
		await expect(page.getByTestId('single-view')).toBeVisible();
		
		// Press Escape
		await page.keyboard.press('Escape');
		
		// Should return to drop zone (removes the single file)
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
	});

	test('should go back to list view with Escape when multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for compression
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Click on job to go to single view
		await page.getByTestId('job-row').first().click();
		await expect(page.getByTestId('single-view')).toBeVisible();
		
		// Press Escape
		await page.keyboard.press('Escape');
		
		// Should return to list view (not drop zone)
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'list');
		await expect(page.getByTestId('list-view')).toBeVisible();
	});
});

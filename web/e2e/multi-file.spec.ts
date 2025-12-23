import { test, expect, FIXTURES } from './fixtures';

test.describe('Multi-File Handling', () => {
	test('should switch to list view when uploading multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files at once
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Should switch to list view
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'list');
		await expect(page.getByTestId('list-view')).toBeVisible();
		
		// Wait for compression to complete
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Should show correct file count
		await expect(page.getByTestId('file-count')).toHaveText('2 files');
	});

	test('should display all jobs in list view', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for compression
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Should have 2 job rows
		const jobRows = page.getByTestId('job-row');
		await expect(jobRows).toHaveCount(2);
		
		// Verify job names are present
		const jobNames = page.getByTestId('job-name');
		const names = await jobNames.allTextContents();
		expect(names).toContain('playground.png');
		expect(names).toContain('multi-agent.jpg');
	});

	test('should show compression status for all jobs', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for all compressions to complete (all jobs should have status 'done')
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// All job rows should eventually show 'done' status
		const jobRows = page.getByTestId('job-row');
		for (const row of await jobRows.all()) {
			await expect(row).toHaveAttribute('data-job-status', 'done', { timeout: 60000 });
		}
		
		// Each job should show savings
		const savings = page.getByTestId('job-savings');
		await expect(savings).toHaveCount(2);
	});

	test('should navigate from list view to single view', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for compression
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Click on the first job
		const firstJob = page.getByTestId('job-row').first();
		await firstJob.click();
		
		// Should switch to single view
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'single');
		await expect(page.getByTestId('single-view')).toBeVisible();
		
		// Back button should be visible (since we have multiple jobs)
		await expect(page.getByTestId('back-button')).toBeVisible();
	});

	test('should navigate back from single view to list view', async ({ page, waitForWasm }) => {
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
		
		// Click back button
		await page.getByTestId('back-button').click();
		
		// Should return to list view
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'list');
		await expect(page.getByTestId('list-view')).toBeVisible();
	});

	test('should offer zip download for multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for compression
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Download button should offer zip
		await expect(page.getByTestId('download-button')).toHaveText('Download All (.zip)');
	});

	test('should trigger zip download for multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for compression
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Set up download listener
		const downloadPromise = page.waitForEvent('download', { timeout: 30000 });
		
		// Click download
		await page.getByTestId('download-button').click();
		
		// Verify zip download
		const download = await downloadPromise;
		expect(download.suggestedFilename()).toBe('compressed-images.zip');
	});

	test('should clear all files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for list view
		await expect(page.getByTestId('list-view')).toBeVisible();
		
		// Click clear all
		await page.getByTestId('clear-all-button').click();
		
		// Should return to drop zone
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
		await expect(page.getByTestId('drop-zone')).toBeVisible();
	});

	test('should show aggregate stats for multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		
		// Upload both files
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		
		// Wait for compression
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		
		// Stats should show aggregate values
		await expect(page.getByTestId('total-original-size')).toBeVisible();
		await expect(page.getByTestId('total-compressed-size')).toBeVisible();
		await expect(page.getByTestId('total-savings')).toBeVisible();
	});
});

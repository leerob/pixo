import { defineConfig, devices } from 'playwright/test';

/**
 * Playwright configuration for E2E smoke tests
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
	testDir: './e2e',
	
	/* Run tests in files in parallel */
	fullyParallel: true,
	
	/* Fail the build on CI if you accidentally left test.only in the source code */
	forbidOnly: !!process.env.CI,
	
	/* Retry on CI to handle flakiness, fewer retries locally */
	retries: process.env.CI ? 2 : 1,
	
	/* Limit parallel workers on CI for stability */
	workers: process.env.CI ? 1 : undefined,
	
	/* Reporter configuration */
	reporter: process.env.CI 
		? [['html', { open: 'never' }], ['github']]
		: [['html', { open: 'on-failure' }]],
	
	/* Shared settings for all projects */
	use: {
		/* Base URL for navigation */
		baseURL: process.env.BASE_URL || 'http://localhost:4173',
		
		/* Collect trace on first retry for debugging */
		trace: 'on-first-retry',
		
		/* Take screenshot on failure */
		screenshot: 'only-on-failure',
		
		/* Record video on first retry */
		video: 'on-first-retry',
		
		/* Reasonable timeout for actions */
		actionTimeout: 10000,
		
		/* Navigation timeout */
		navigationTimeout: 30000,
	},
	
	/* Global timeout for each test */
	timeout: 60000,
	
	/* Expect timeout */
	expect: {
		timeout: 10000,
	},

	/* Configure projects for browsers */
	projects: [
		{
			name: 'chromium',
			use: { 
				...devices['Desktop Chrome'],
				viewport: { width: 1280, height: 720 },
			},
		},
	],

	/* Run a local preview server before running tests */
	webServer: {
		command: 'npm run preview',
		url: 'http://localhost:4173',
		reuseExistingServer: !process.env.CI,
		timeout: 120000,
	},
});

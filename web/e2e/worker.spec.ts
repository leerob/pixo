import { test, expect, FIXTURES } from "./fixtures";

test.describe("Web Worker Compression", () => {
  test("should compress large PNG without freezing UI", async ({
    page,
    waitForWasm,
  }) => {
    await page.goto("/");
    await waitForWasm();

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(FIXTURES.PNG);

    await expect(page.getByTestId("main-content")).toHaveAttribute(
      "data-view-mode",
      "single",
      { timeout: 30000 },
    );
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();
  });

  test("should handle PNG preset changes without blocking", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    const slider = page.getByTestId("png-preset-slider");
    await slider.fill("0"); // Smaller preset

    await page.waitForTimeout(500);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible({
      timeout: 30000,
    });
  });

  test("should compress at 4x zoom without timeout", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    await page.getByTestId("zoom-4x").click();

    const slider = page.getByTestId("png-preset-slider");
    await slider.fill("0");

    await page.waitForTimeout(500);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible({
      timeout: 60000,
    });
  });

  test("should handle resize with worker", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    await page.getByRole("checkbox", { name: "Resize" }).click();

    // Wait for resize controls and fill them
    const widthInput = page.locator('input[type="number"]').first();
    const heightInput = page.locator('input[type="number"]').nth(1);

    await expect(widthInput).toBeVisible();
    await widthInput.fill("800");
    await heightInput.fill("600");

    await page.getByRole("button", { name: "Resize" }).click();

    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible({
      timeout: 30000,
    });
  });
});

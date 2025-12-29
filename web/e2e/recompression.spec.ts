import { test, expect, FIXTURES } from "./fixtures";

test.describe("Recompression Tests", () => {
  test("should recompress when changing PNG preset", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    const slider = page.getByTestId("png-preset-slider");
    await slider.fill("2"); // Change to Faster
    await slider.blur(); // Trigger onchange

    // Wait for recompression to complete by checking button is not disabled
    await expect(page.getByTestId("download-button")).toBeEnabled({
      timeout: 30000,
    });
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();
  });

  test("should recompress when changing JPEG quality", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.JPEG);

    const initialSize = await page
      .getByTestId("total-compressed-size")
      .textContent();

    const qualitySlider = page.getByTestId("quality-slider");
    await qualitySlider.fill("70");
    await page.waitForTimeout(1000);

    const newSize = await page
      .getByTestId("total-compressed-size")
      .textContent();
    expect(newSize).not.toBe(initialSize);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();
  });

  test("should recompress when toggling lossless", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    const initialSize = await page
      .getByTestId("total-compressed-size")
      .textContent();

    await page.getByRole("checkbox", { name: "Lossless" }).click();
    await page.waitForTimeout(1000);

    const newSize = await page
      .getByTestId("total-compressed-size")
      .textContent();
    expect(newSize).not.toBe(initialSize);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();
  });

  test("should handle multiple recompressions without error", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    const slider = page.getByTestId("png-preset-slider");

    // Multiple preset changes
    await slider.fill("0");
    await page.waitForTimeout(500);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();

    await slider.fill("2");
    await page.waitForTimeout(500);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();

    await slider.fill("1");
    await page.waitForTimeout(500);
    await expect(page.getByTestId("compressed-image-overlay")).toBeVisible();

    // Should not show error
    await expect(page.getByTestId("single-view")).toBeVisible();
  });

  test("should preserve original ImageData after compression", async ({
    page,
    waitForWasm,
    uploadAndWaitForCompression,
  }) => {
    await page.goto("/");
    await waitForWasm();
    await uploadAndWaitForCompression(FIXTURES.PNG);

    const originalDimensions = await page
      .getByTestId("image-dimensions")
      .textContent();

    // Change preset multiple times
    const slider = page.getByTestId("png-preset-slider");
    await slider.fill("0");
    await page.waitForTimeout(1000);
    await slider.fill("2");
    await page.waitForTimeout(1000);

    // Dimensions should remain the same
    const currentDimensions = await page
      .getByTestId("image-dimensions")
      .textContent();
    expect(currentDimensions).toBe(originalDimensions);
  });
});

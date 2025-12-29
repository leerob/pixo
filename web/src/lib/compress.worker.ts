import { compressImage, resizeImage } from "./wasm";
import type { CompressOptions } from "./wasm";

interface CompressMessage {
  id: string;
  type: "compress";
  width: number;
  height: number;
  data: ArrayBuffer;
  options: CompressOptions;
}

interface ResizeMessage {
  id: string;
  type: "resize";
  width: number;
  height: number;
  data: ArrayBuffer;
  options: {
    width: number;
    height: number;
    algorithm: "nearest" | "bilinear" | "lanczos3";
    maintainAspectRatio: boolean;
  };
}

type WorkerMessage = CompressMessage | ResizeMessage;

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const { id, type } = e.data;

  try {
    if (type === "compress") {
      const { width, height, data, options } = e.data;
      const imageData = new ImageData(
        new Uint8ClampedArray(data),
        width,
        height,
      );
      const result = await compressImage(imageData, options);
      self.postMessage({
        id,
        success: true,
        result: {
          blob: result.blob,
          elapsedMs: result.elapsedMs,
        },
      });
    } else if (type === "resize") {
      const { width, height, data, options } = e.data;
      const imageData = new ImageData(
        new Uint8ClampedArray(data),
        width,
        height,
      );
      const result = await resizeImage(imageData, options);
      const resultBuffer = result.data.buffer;
      self.postMessage(
        {
          id,
          success: true,
          result: {
            width: result.width,
            height: result.height,
            data: resultBuffer,
          },
        },
        { transfer: [resultBuffer] },
      );
    }
  } catch (error) {
    self.postMessage({
      id,
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
};

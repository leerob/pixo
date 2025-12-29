import type { CompressOptions, CompressResult, ResizeOptions } from "./wasm";

let worker: Worker | null = null;
let messageId = 0;
const pendingRequests = new Map<
  string,
  {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: ReturnType<typeof setTimeout>;
  }
>();

const REQUEST_TIMEOUT = 120000; // 2 minutes max per operation

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./compress.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (e: MessageEvent) => {
      const { id, success, result, error } = e.data;
      const pending = pendingRequests.get(id);
      if (pending) {
        clearTimeout(pending.timeout);
        pendingRequests.delete(id);
        if (success) {
          if (result.data instanceof ArrayBuffer) {
            const reconstructed = new ImageData(
              new Uint8ClampedArray(result.data),
              result.width,
              result.height,
            );
            pending.resolve(reconstructed);
          } else {
            pending.resolve(result);
          }
        } else {
          pending.reject(new Error(error));
        }
      }
    };
    worker.onerror = (err) => {
      console.error("Worker error:", err);
      pendingRequests.forEach(({ reject, timeout }) => {
        clearTimeout(timeout);
        reject(new Error("Worker crashed"));
      });
      pendingRequests.clear();
      worker = null;
    };
  }
  return worker;
}

export async function compressImage(
  imageData: ImageData,
  options: CompressOptions,
): Promise<CompressResult> {
  return new Promise((resolve, reject) => {
    const id = `compress-${++messageId}`;
    const timeout = setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error("Compression timeout"));
    }, REQUEST_TIMEOUT);

    pendingRequests.set(id, { resolve, reject, timeout });
    const buffer = imageData.data.buffer.slice(0);
    getWorker().postMessage(
      {
        id,
        type: "compress",
        width: imageData.width,
        height: imageData.height,
        data: buffer,
        options,
      },
      { transfer: [buffer] },
    );
  });
}

export async function resizeImage(
  imageData: ImageData,
  options: ResizeOptions,
): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const id = `resize-${++messageId}`;
    const timeout = setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error("Resize timeout"));
    }, REQUEST_TIMEOUT);

    pendingRequests.set(id, { resolve, reject, timeout });
    const buffer = imageData.data.buffer.slice(0);
    getWorker().postMessage(
      {
        id,
        type: "resize",
        width: imageData.width,
        height: imageData.height,
        data: buffer,
        options,
      },
      { transfer: [buffer] },
    );
  });
}

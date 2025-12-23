export type EncodeFormat = 'png' | 'jpeg';
export type PngFilter =
	| 'adaptive'
	| 'adaptive-fast'
	| 'none'
	| 'sub'
	| 'up'
	| 'average'
	| 'paeth';

export type CompressOptions = {
	format: EncodeFormat;
	quality?: number; // JPEG quality 1-100
	compressionLevel?: number; // PNG compression level 1-9
	filter?: PngFilter;
	subsampling420?: boolean;
	hasAlpha?: boolean; // If false, strip alpha channel for smaller output
};

export type CompressResult = {
	bytes: Uint8Array;
	blob: Blob;
	elapsedMs: number;
};

// Dynamically imported WASM module (client-side only)
let wasmModule: typeof import('$lib/comprs-wasm/comprs.js') | null = null;
let initialized: Promise<void> | null = null;

export function initWasm() {
	if (!initialized) {
		initialized = import('$lib/comprs-wasm/comprs.js').then(async (mod) => {
			await mod.default();
			wasmModule = mod;
		});
	}
	return initialized;
}

const filterMap: Record<PngFilter, number> = {
	none: 0,
	sub: 1,
	up: 2,
	average: 3,
	paeth: 4,
	adaptive: 5,
	'adaptive-fast': 6
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

function rgbaToRgb(data: Uint8ClampedArray) {
	const rgb = new Uint8Array((data.length / 4) * 3);
	for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
		rgb[j] = data[i];
		rgb[j + 1] = data[i + 1];
		rgb[j + 2] = data[i + 2];
	}
	return rgb;
}

export async function compressImage(imageData: ImageData, options: CompressOptions): Promise<CompressResult> {
	await initWasm();
	if (!wasmModule) throw new Error('WASM module not loaded');

	const t0 = performance.now();
	let bytes: Uint8Array;
	let mime: string;

	if (options.format === 'png') {
		const compressionLevel = clamp(options.compressionLevel ?? 6, 1, 9);
		const filterCode = filterMap[options.filter ?? 'adaptive'];

		// If the image has no meaningful alpha, encode as RGB (color_type 2) for smaller output
		// This avoids the overhead of encoding an all-255 alpha channel
		const useRgb = options.hasAlpha === false;

		if (useRgb) {
			const rgb = rgbaToRgb(imageData.data);
			bytes = wasmModule.encodePngWithFilter(rgb, imageData.width, imageData.height, 2, compressionLevel, filterCode);
		} else {
			const rgba = new Uint8Array(imageData.data);
			bytes = wasmModule.encodePngWithFilter(rgba, imageData.width, imageData.height, 3, compressionLevel, filterCode);
		}
		mime = 'image/png';
	} else {
		const quality = clamp(options.quality ?? 85, 1, 100);
		const rgb = rgbaToRgb(imageData.data);
		bytes = wasmModule.encodeJpeg(rgb, imageData.width, imageData.height, quality, 2, options.subsampling420 ?? true);
		mime = 'image/jpeg';
	}

	const elapsedMs = performance.now() - t0;
	const blob = new Blob([new Uint8Array(bytes)], { type: mime });
	return { bytes, blob, elapsedMs };
}

export async function getBytesPerPixel(color: number) {
	await initWasm();
	if (!wasmModule) throw new Error('WASM module not loaded');
	return wasmModule.bytesPerPixel(color);
}

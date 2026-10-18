// Reads an animated GIF's frame count and speed straight from its blocks, without decoding pixels.

export interface GifInfo {
  frames: number;
  /** Average frames per second, as browsers play it. */
  fps: number;
}

export function gifInfo(data: ArrayBuffer): GifInfo | null {
  const b = new Uint8Array(data);
  const text = String.fromCharCode(...b.subarray(0, 6));
  if (text !== "GIF89a" && text !== "GIF87a") return null;
  let i = 13;
  if (b[10] & 0x80) i += 3 * 2 ** ((b[10] & 7) + 1);
  const skipSubBlocks = () => {
    while (i < b.length && b[i] !== 0) i += b[i] + 1;
    i++;
  };
  let frames = 0;
  let hundredths = 0;
  let delay = 0;
  while (i < b.length) {
    const block = b[i++];
    if (block === 0x3b) break;
    if (block === 0x21) {
      const label = b[i++];
      // Browsers play delays under 2/100 s as 1/10 s.
      if (label === 0xf9) delay = b[i + 2] | (b[i + 3] << 8);
      skipSubBlocks();
    } else if (block === 0x2c) {
      const packed = b[i + 8];
      i += 9;
      if (packed & 0x80) i += 3 * 2 ** ((packed & 7) + 1);
      i++; // LZW code size
      skipSubBlocks();
      frames++;
      hundredths += delay < 2 ? 10 : delay;
      delay = 0;
    } else {
      break;
    }
  }
  if (!frames) return null;
  return { frames, fps: Math.round((frames * 100 * 100) / hundredths) / 100 };
}

/** Second opinion from the browser's own decoder (Chrome, Edge, newer Firefox), for GIFs the block reader can't follow. */
export async function decodeGifInfo(file: Blob): Promise<GifInfo | null> {
  const Decoder = (globalThis as { ImageDecoder?: new (init: { data: ReadableStream; type: string }) => any }).ImageDecoder;
  if (!Decoder) return null;
  try {
    const decoder = new Decoder({ data: file.stream(), type: "image/gif" });
    await decoder.tracks.ready;
    const frames: number = decoder.tracks.selectedTrack.frameCount;
    let micros = 0;
    for (let i = 0; i < frames; i++) {
      const { image } = await decoder.decode({ frameIndex: i });
      micros += image.duration || 100_000;
      image.close();
    }
    decoder.close();
    return frames > 1 ? { frames, fps: Math.round((frames * 1e8) / micros) / 100 } : null;
  } catch {
    return null;
  }
}

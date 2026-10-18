import { describe, expect, it } from "vitest";
import { gifInfo } from "./gif";

/** A tiny GIF: 2x2, global palette, `delays` frames (in 1/100 s), each one LZW block. */
export function makeGif(delays: number[]): ArrayBuffer {
  const bytes = [...Array.from("GIF89a", (c) => c.charCodeAt(0)), 2, 0, 2, 0, 0x80, 0, 0, 0, 0, 0, 255, 255, 255];
  bytes.push(0x21, 0xff, 11, ...Array.from("NETSCAPE2.0", (c) => c.charCodeAt(0)), 3, 1, 0, 0, 0);
  for (const d of delays) {
    bytes.push(0x21, 0xf9, 4, 0, d & 255, d >> 8, 0, 0);
    bytes.push(0x2c, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 2, 0x44, 0x01, 0);
  }
  bytes.push(0x3b);
  return new Uint8Array(bytes).buffer;
}

describe("gifInfo", () => {
  it("counts frames and works out the speed", () => {
    expect(gifInfo(makeGif([10, 10, 10, 10]))).toEqual({ frames: 4, fps: 10 });
    expect(gifInfo(makeGif([4, 4, 4]))).toEqual({ frames: 3, fps: 25 });
    expect(gifInfo(makeGif([0, 0]))).toEqual({ frames: 2, fps: 10 });
  });

  it("ignores files that aren't GIFs", () => {
    expect(gifInfo(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0, 0, 0]).buffer)).toBeNull();
  });
});

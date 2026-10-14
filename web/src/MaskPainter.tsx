import { useEffect, useRef, useState } from "react";

/**
 * Paint over the part of a picture to repaint. Returns the picture with that area made
 * transparent, which ComfyUI's LoadImage reads as the inpaint mask.
 */
export function MaskPainter(props: { src: string; onDone: (masked: Blob) => void; onCancel: () => void }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const paintRef = useRef<HTMLCanvasElement>(null);
  const [brush, setBrush] = useState(40);
  const [ready, setReady] = useState(false);
  const drawing = useRef(false);

  useEffect(() => {
    const img = imgRef.current!;
    const fit = () => {
      const c = paintRef.current!;
      c.width = img.clientWidth;
      c.height = img.clientHeight;
      setReady(true);
    };
    if (img.complete) fit();
    else img.onload = fit;
  }, [props.src]);

  function paint(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!drawing.current) return;
    const c = paintRef.current!;
    const rect = c.getBoundingClientRect();
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "rgb(255, 60, 60)";
    ctx.beginPath();
    ctx.arc(e.clientX - rect.left, e.clientY - rect.top, brush / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function done() {
    const img = imgRef.current!;
    const out = document.createElement("canvas");
    out.width = img.naturalWidth;
    out.height = img.naturalHeight;
    const ctx = out.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    // Cut the painted area out of the picture: transparent pixels become the mask.
    ctx.globalCompositeOperation = "destination-out";
    ctx.drawImage(paintRef.current!, 0, 0, out.width, out.height);
    out.toBlob((b) => b && props.onDone(b), "image/png");
  }

  const clear = () => {
    const c = paintRef.current!;
    c.getContext("2d")!.clearRect(0, 0, c.width, c.height);
  };

  return (
    <div className="modal" role="dialog" aria-label="Paint the area to change">
      <div className="modal-body">
        <p className="small muted">Paint over the part you want to change. Everything else stays as it is.</p>
        <div className="paint">
          <img ref={imgRef} src={props.src} alt="" draggable={false} />
          <canvas
            ref={paintRef}
            onPointerDown={(e) => {
              drawing.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              paint(e);
            }}
            onPointerMove={paint}
            onPointerUp={() => (drawing.current = false)}
          />
        </div>
        <div className="row">
          <label className="row small">
            Brush
            <input type="range" min={8} max={160} value={brush} onChange={(e) => setBrush(Number(e.target.value))} />
          </label>
          <button className="link" onClick={clear}>Clear</button>
          <span style={{ flex: 1 }} />
          <button className="link" onClick={props.onCancel}>Cancel</button>
          <button className="primary" disabled={!ready} onClick={done}>Done</button>
        </div>
      </div>
    </div>
  );
}

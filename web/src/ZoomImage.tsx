import { useEffect, useRef, useState } from "react";

/** The preview picture: scroll to zoom at the pointer, drag to move, double-click to fit again. */
export function ZoomImage(props: { src: string; alt: string }) {
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => setView({ scale: 1, x: 0, y: 0 }), [props.src]);

  // React's onWheel is passive, so the page would scroll too; listen directly instead.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      // Pointer position relative to the box centre, where the scale is anchored.
      const px = e.clientX - r.left - r.width / 2;
      const py = e.clientY - r.top - r.height / 2;
      setView((v) => {
        const scale = Math.min(8, Math.max(1, v.scale * Math.exp(-e.deltaY * 0.0015)));
        if (scale === 1) return { scale: 1, x: 0, y: 0 };
        const k = scale / v.scale;
        return { scale, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div
      ref={box}
      className={view.scale > 1 ? "zoom zoomed" : "zoom"}
      onPointerDown={(e) => {
        if (view.scale === 1) return;
        drag.current = { x: e.clientX - view.x, y: e.clientY - view.y };
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => drag.current && setView((v) => ({ ...v, x: e.clientX - drag.current!.x, y: e.clientY - drag.current!.y }))}
      onPointerUp={() => (drag.current = null)}
      onDoubleClick={() => setView({ scale: 1, x: 0, y: 0 })}
    >
      <img
        src={props.src}
        alt={props.alt}
        draggable={false}
        style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}
      />
      {view.scale > 1 && <span className="zoom-level small">{Math.round(view.scale * 100)}%</span>}
    </div>
  );
}

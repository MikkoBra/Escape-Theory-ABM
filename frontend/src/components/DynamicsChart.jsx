import { useEffect, useRef, useCallback } from "react";

const FONT_SM   = "11px 'Source Code Pro', monospace";
const FONT_XS   = "10px 'Source Sans 3', sans-serif";
const GRID_COL  = "rgba(0,0,0,0.07)";
const AXIS_COL  = "rgba(0,0,0,0.35)";
const LABEL_COL = "rgba(0,0,0,0.45)";

export default function DynamicsChart({ series = [], dt = 1/24/60 }) {
  const canvasRef    = useRef(null);
  const crosshairRef = useRef(null);
  const overlayRef   = useRef(null);

  const pad = { top: 20, right: 28, bottom: 52, left: 58 };

  // Read logical size from the *parent* before touching canvas dimensions,
  // so the canvas never influences its own measurement.
  function getContainerSize(canvas) {
    const parent = canvas.parentElement;
    return { w: parent.clientWidth, h: parent.clientHeight };
  }

  const drawChart = useCallback(() => {
    const canvas    = canvasRef.current;
    const crosshair = crosshairRef.current;
    if (!canvas || series.length === 0) return;

    const length = series[0].data.length;
    if (length < 2) return;

    const dpr      = window.devicePixelRatio || 1;
    const { w, h } = getContainerSize(canvas);

    // Size both canvases together, once, here — never in mouse handlers
    [canvas, crosshair].forEach(c => {
      if (!c) return;
      c.width  = w * dpr;
      c.height = h * dpr;
      c.style.width  = w + "px";
      c.style.height = h + "px";
    });

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top  - pad.bottom;

    // ── grid ─────────────────────────────────────────────
    const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
    yTicks.forEach(val => {
      const y = pad.top + (1 - val) * chartH;
      ctx.strokeStyle = GRID_COL;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + chartW, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = LABEL_COL;
      ctx.font = FONT_SM;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(val.toFixed(2), pad.left - 8, y);
    });

    // ── x-axis ───────────────────────────────────────────
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    const nX = 6;
    for (let i = 0; i <= nX; i++) {
      const idx  = Math.round((length - 1) * (i / nX));
      const days = (idx * dt).toFixed(2);
      const x    = pad.left + (idx / (length - 1)) * chartW;

      ctx.strokeStyle = AXIS_COL;
      ctx.beginPath();
      ctx.moveTo(x, pad.top + chartH);
      ctx.lineTo(x, pad.top + chartH + 4);
      ctx.stroke();

      ctx.fillStyle = LABEL_COL;
      ctx.font = FONT_SM;
      ctx.fillText(days, x, pad.top + chartH + 7);
    }

    // ── axes ─────────────────────────────────────────────
    ctx.strokeStyle = AXIS_COL;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + chartH);
    ctx.lineTo(pad.left + chartW, pad.top + chartH);
    ctx.stroke();

    // ── labels ───────────────────────────────────────────
    ctx.fillStyle = LABEL_COL;
    ctx.font = FONT_XS;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("Time (days)", pad.left + chartW / 2, h - 2);

    // ── draw all series ──────────────────────────────────
    series.forEach(s => {
      ctx.beginPath();
      s.data.forEach((val, i) => {
        const x = pad.left + (i / (length - 1)) * chartW;
        const y = pad.top  + (1 - val) * chartH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

  }, [series, dt]);

  useEffect(() => {
    drawChart();
    const ro = new ResizeObserver(() => drawChart());
    const parent = canvasRef.current?.parentElement;
    if (parent) ro.observe(parent);
    return () => ro.disconnect();
  }, [drawChart]);

  // ── hover ─────────────────────────────────────────────
  function handleMouseMove(e) {
    const canvas    = canvasRef.current;
    const crosshair = crosshairRef.current;
    const overlay   = overlayRef.current;
    if (!canvas || !crosshair || !overlay || series.length === 0) return;

    const length = series[0].data.length;
    if (length < 2) return;

    const rect   = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const w      = canvas.offsetWidth;
    const h      = canvas.offsetHeight;
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top  - pad.bottom;

    if (
      mouseX < pad.left || mouseX > pad.left + chartW ||
      mouseY < pad.top  || mouseY > pad.top  + chartH
    ) {
      overlay.style.display = "none";
      const octx = crosshair.getContext("2d");
      octx.clearRect(0, 0, crosshair.width, crosshair.height);
      return;
    }

    const frac = (mouseX - pad.left) / chartW;
    const idx  = Math.round(frac * (length - 1));
    const days = (idx * dt).toFixed(3);

    const ROW = `style="display:flex;justify-content:space-between;gap:16px;"`;
    const SEC = `style="color:rgba(0,0,0,0.45);"`;
    let html = `
      <div ${ROW}><span ${SEC}>Step</span><span>${idx}</span></div>
      <div ${ROW}><span ${SEC}>Time</span><span>${days} d</span></div>
    `;
    series.forEach(s => {
      const val = s.data[idx] ?? 0;
      html += `
        <div ${ROW}>
          <span ${SEC}>${s.name}</span>
          <span style="color:${s.color}">${val.toFixed(4)}</span>
        </div>
      `;
    });

    overlay.style.display = "block";
    overlay.style.left    = (mouseX + 12) + "px";
    overlay.style.top     = (mouseY - 20) + "px";
    overlay.innerHTML     = html;

    // crosshair — canvas is already sized; just clear + redraw
    const dpr  = window.devicePixelRatio || 1;
    const octx = crosshair.getContext("2d");
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.clearRect(0, 0, w, h);

    const cx = pad.left + (idx / (length - 1)) * chartW;

    octx.setLineDash([3, 3]);
    octx.beginPath();
    octx.moveTo(cx, pad.top);
    octx.lineTo(cx, pad.top + chartH);
    octx.strokeStyle = "rgba(0,0,0,0.2)";
    octx.lineWidth   = 1;
    octx.stroke();
    octx.setLineDash([]);

    series.forEach(s => {
      const val = s.data[idx] ?? 0;
      const cy  = pad.top + (1 - val) * chartH;
      octx.beginPath();
      octx.arc(cx, cy, 4, 0, Math.PI * 2);
      octx.fillStyle = s.color;
      octx.fill();
    });
  }

  function handleMouseLeave() {
    const overlay   = overlayRef.current;
    const crosshair = crosshairRef.current;
    if (overlay) overlay.style.display = "none";
    if (crosshair) {
      const octx = crosshair.getContext("2d");
      octx.clearRect(0, 0, crosshair.width, crosshair.height);
    }
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <canvas
        ref={canvasRef}
        className="chart-canvas"
        style={{ position: "absolute", inset: 0 }}
      />
      <canvas
        ref={crosshairRef}
        className="chart-crosshair"
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      />
      <div
        style={{ position: "absolute", inset: 0 }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      <div
        ref={overlayRef}
        style={{
          display: "none",
          position: "absolute",
          pointerEvents: "none",
          background: "rgba(255,255,255,0.92)",
          border: "1px solid rgba(0,0,0,0.1)",
          borderRadius: "6px",
          padding: "6px 10px",
          fontSize: "11px",
          fontFamily: "'Source Code Pro', monospace",
          color: "rgba(0,0,0,0.75)",
          whiteSpace: "nowrap",
          zIndex: 10,
        }}
      />
    </div>
  );
}
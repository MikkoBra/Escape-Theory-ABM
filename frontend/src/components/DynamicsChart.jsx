import { useEffect, useRef, useCallback } from "react";

const FONT_SM   = "11px 'Source Code Pro', monospace";
const FONT_XS   = "10px 'Source Sans 3', sans-serif";
const LINE_COL  = "#2a5c8a";
const GRID_COL  = "rgba(0,0,0,0.07)";
const AXIS_COL  = "rgba(0,0,0,0.35)";
const LABEL_COL = "rgba(0,0,0,0.45)";

export default function DynamicsChart({ data, dt = 1/24/60 }) {
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);

  const pad = { top: 20, right: 28, bottom: 52, left: 58 };

  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;

    canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;

    const ctx = canvas.getContext("2d");
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    ctx.clearRect(0, 0, w, h);

    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top  - pad.bottom;

    // ── grid + y labels ───────────────────────────────────────
    const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
    yTicks.forEach(val => {
      const y = pad.top + (1 - val) * chartH;
      ctx.strokeStyle = GRID_COL;
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + chartW, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle    = LABEL_COL;
      ctx.font         = FONT_SM;
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(val.toFixed(2), pad.left - 8, y);
    });

    // ── x labels ──────────────────────────────────────────────
    ctx.textAlign    = "center";
    ctx.textBaseline = "top";
    const nX = 6;
    for (let i = 0; i <= nX; i++) {
      const idx  = Math.round((data.length - 1) * (i / nX));
      const days = (idx * dt).toFixed(2);
      const x    = pad.left + (idx / (data.length - 1)) * chartW;
      ctx.strokeStyle = AXIS_COL;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(x, pad.top + chartH);
      ctx.lineTo(x, pad.top + chartH + 4);
      ctx.stroke();
      ctx.fillStyle = LABEL_COL;
      ctx.font      = FONT_SM;
      ctx.fillText(days, x, pad.top + chartH + 7);
    }

    // ── axis lines ────────────────────────────────────────────
    ctx.strokeStyle = AXIS_COL;
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + chartH);
    ctx.lineTo(pad.left + chartW, pad.top + chartH);
    ctx.stroke();

    // ── axis labels ───────────────────────────────────────────
    ctx.fillStyle    = LABEL_COL;
    ctx.font         = FONT_XS;
    ctx.textAlign    = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("Time (days)", pad.left + chartW / 2, h - 2);

    ctx.save();
    ctx.translate(13, pad.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "top";
    ctx.fillText("Stress  S(t)", 0, 0);
    ctx.restore();

    // ── line ──────────────────────────────────────────────────
    ctx.beginPath();
    data.forEach((val, i) => {
      const x = pad.left + (i / (data.length - 1)) * chartW;
      const y = pad.top  + (1 - val) * chartH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = LINE_COL;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = "round";
    ctx.stroke();

  }, [data, dt]);

  useEffect(() => { drawChart(); }, [drawChart]);

  // ── hover handler ─────────────────────────────────────────────
  function handleMouseMove(e) {
    const canvas  = canvasRef.current;
    const overlay = overlayRef.current;
    if (!canvas || !overlay || data.length < 2) return;

    const rect   = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const w      = canvas.offsetWidth;
    const h      = canvas.offsetHeight;
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top  - pad.bottom;

    // outside chart area
    if (
      mouseX < pad.left || mouseX > pad.left + chartW ||
      mouseY < pad.top  || mouseY > pad.top  + chartH
    ) {
      overlay.style.display = "none";
      return;
    }

    const frac  = (mouseX - pad.left) / chartW;
    const idx   = Math.round(frac * (data.length - 1));
    const val   = data[Math.max(0, Math.min(idx, data.length - 1))];
    const days  = (idx * dt).toFixed(3);

    // position overlay — flip left if near right edge
    const BOX_W = 130;
    const left  = mouseX + 12 + BOX_W > w ? mouseX - BOX_W - 12 : mouseX + 12;
    const top   = Math.max(pad.top, mouseY - 20);

    overlay.style.display = "block";
    overlay.style.left    = left + "px";
    overlay.style.top     = top  + "px";
    overlay.innerHTML     = `
      <div class="hover-row"><span>Step</span><span>${idx}</span></div>
      <div class="hover-row"><span>Time</span><span>${days} d</span></div>
      <div class="hover-row accent"><span>S(t)</span><span>${val.toFixed(4)}</span></div>
    `;

    // crosshair on overlay canvas
    const oc  = overlayRef.current.parentElement.querySelector(".chart-crosshair");
    if (!oc) return;
    oc.width  = canvas.width;
    oc.height = canvas.height;
    const octx = oc.getContext("2d");
    octx.scale(window.devicePixelRatio, window.devicePixelRatio);
    octx.clearRect(0, 0, w, h);

    const cx = pad.left + (idx / (data.length - 1)) * chartW;
    const cy = pad.top  + (1 - val) * chartH;

    octx.strokeStyle = "rgba(42,92,138,0.3)";
    octx.lineWidth   = 1;
    octx.setLineDash([3, 3]);
    octx.beginPath();
    octx.moveTo(cx, pad.top);
    octx.lineTo(cx, pad.top + chartH);
    octx.stroke();
    octx.beginPath();
    octx.moveTo(pad.left, cy);
    octx.lineTo(pad.left + chartW, cy);
    octx.stroke();
    octx.setLineDash([]);

    octx.beginPath();
    octx.arc(cx, cy, 4, 0, Math.PI * 2);
    octx.fillStyle   = LINE_COL;
    octx.fill();
    octx.strokeStyle = "#fff";
    octx.lineWidth   = 1.5;
    octx.stroke();
  }

  function handleMouseLeave() {
    const overlay = overlayRef.current;
    if (overlay) overlay.style.display = "none";
    const oc = overlayRef.current?.parentElement?.querySelector(".chart-crosshair");
    if (oc) oc.getContext("2d").clearRect(0, 0, oc.width, oc.height);
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <canvas ref={canvasRef} className="chart-canvas" />
      <canvas
        className="chart-canvas chart-crosshair"
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {/* invisible hit area on top */}
      <div
        style={{ position: "absolute", inset: 0 }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      <div
        ref={overlayRef}
        className="chart-tooltip"
        style={{ display: "none" }}
      />
      <style>{`
        .chart-tooltip {
          position: absolute;
          background: #fff;
          border: 1px solid var(--border);
          border-radius: 3px;
          padding: 8px 10px;
          pointer-events: none;
          font-family: 'Source Code Pro', monospace;
          font-size: 11px;
          color: var(--text);
          box-shadow: 0 2px 8px rgba(0,0,0,0.08);
          min-width: 120px;
          z-index: 10;
        }
        .hover-row {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          color: var(--text-muted);
          line-height: 1.8;
        }
        .hover-row span:last-child { color: var(--text); }
        .hover-row.accent span:last-child {
          color: var(--accent);
          font-weight: 500;
        }
      `}</style>
    </div>
  );
}
import { useMemo } from "react";

interface Props {
  layers: number[];
  weights: number[][][];
  width: number;
  height: number;
}

function weightColor(w: number): string {
  // Blue for positive, Red for negative.
  const a = Math.min(1, Math.abs(w) / 1.5);
  if (w >= 0) {
    // bright cyan-blue
    return `rgba(56, 189, 248, ${0.15 + a * 0.85})`;
  }
  return `rgba(248, 113, 113, ${0.15 + a * 0.85})`;
}

function weightStroke(w: number): number {
  return 0.4 + Math.min(4, Math.abs(w) * 2.2);
}

export function NetworkCanvas({ layers, weights, width, height }: Props) {
  const positions = useMemo(() => {
    const padX = 60;
    const padY = 40;
    const cols = layers.length;
    const colW = (width - padX * 2) / Math.max(1, cols - 1);
    const pos: { x: number; y: number }[][] = [];
    for (let l = 0; l < cols; l++) {
      const n = layers[l];
      const rowH = (height - padY * 2) / Math.max(1, n - 1 || 1);
      const layerPos: { x: number; y: number }[] = [];
      for (let i = 0; i < n; i++) {
        const x = padX + l * colW;
        const y = n === 1 ? height / 2 : padY + i * rowH;
        layerPos.push({ x, y });
      }
      pos.push(layerPos);
    }
    return pos;
  }, [layers, width, height]);

  const layerLabels = useMemo(() => {
    return layers.map((_, i) => {
      if (i === 0) return "Input";
      if (i === layers.length - 1) return "Output";
      return `Hidden ${i}`;
    });
  }, [layers]);

  return (
    <svg
      width={width}
      height={height}
      className="block"
      viewBox={`0 0 ${width} ${height}`}
    >
      <defs>
        <radialGradient id="nodeGrad" cx="0.3" cy="0.3" r="0.8">
          <stop offset="0%" stopColor="#475569" />
          <stop offset="100%" stopColor="#1e293b" />
        </radialGradient>
        <radialGradient id="nodeGradInput" cx="0.3" cy="0.3" r="0.8">
          <stop offset="0%" stopColor="#7dd3fc" />
          <stop offset="100%" stopColor="#0369a1" />
        </radialGradient>
        <radialGradient id="nodeGradOutput" cx="0.3" cy="0.3" r="0.8">
          <stop offset="0%" stopColor="#86efac" />
          <stop offset="100%" stopColor="#15803d" />
        </radialGradient>
      </defs>

      {/* layer labels */}
      {positions.map((col, l) => (
        <text
          key={`label-${l}`}
          x={col[0].x}
          y={20}
          textAnchor="middle"
          className="fill-slate-400"
          style={{ fontSize: 11, fontFamily: "Inter, sans-serif" }}
        >
          {layerLabels[l]}
        </text>
      ))}

      {/* edges */}
      {weights.map((W, l) =>
        W.map((row, j) =>
          row.map((w, i) => {
            const from = positions[l][i];
            const to = positions[l + 1][j];
            return (
              <line
                key={`e-${l}-${j}-${i}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={weightColor(w)}
                strokeWidth={weightStroke(w)}
                strokeLinecap="round"
              />
            );
          }),
        ),
      )}

      {/* nodes */}
      {positions.map((col, l) =>
        col.map((p, i) => {
          const isInput = l === 0;
          const isOutput = l === positions.length - 1;
          const fill = isInput
            ? "url(#nodeGradInput)"
            : isOutput
              ? "url(#nodeGradOutput)"
              : "url(#nodeGrad)";
          return (
            <g key={`n-${l}-${i}`}>
              <circle
                cx={p.x}
                cy={p.y}
                r={14}
                fill={fill}
                stroke="#0f172a"
                strokeWidth={2}
              />
              {isInput && (
                <text
                  x={p.x}
                  y={p.y + 4}
                  textAnchor="middle"
                  className="fill-slate-900"
                  style={{ fontSize: 11, fontWeight: 600 }}
                >
                  {i === 0 ? "x" : "y"}
                </text>
              )}
              {isOutput && (
                <text
                  x={p.x}
                  y={p.y + 4}
                  textAnchor="middle"
                  className="fill-slate-900"
                  style={{ fontSize: 11, fontWeight: 600 }}
                >
                  ŷ
                </text>
              )}
            </g>
          );
        }),
      )}
    </svg>
  );
}

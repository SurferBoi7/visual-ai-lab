import { useMemo } from "react";
import type { DataPoint } from "@/lib/nn";

interface Props {
  data: DataPoint[];
  grid: Float32Array | null;
  gridRes: number;
  size: number;
}

export function DataView({ data, grid, gridRes, size }: Props) {
  const cell = size / gridRes;

  const cells = useMemo(() => {
    if (!grid) return null;
    const rects: { x: number; y: number; fill: string }[] = [];
    for (let yi = 0; yi < gridRes; yi++) {
      for (let xi = 0; xi < gridRes; xi++) {
        const v = grid[yi * gridRes + xi];
        const isClass1 = v > 0.5;
        const conf = Math.abs(v - 0.5) * 2; // 0..1
        const alpha = 0.12 + conf * 0.55;
        const fill = isClass1
          ? `rgba(74, 222, 128, ${alpha})`
          : `rgba(56, 189, 248, ${alpha})`;
        rects.push({ x: xi * cell, y: (gridRes - 1 - yi) * cell, fill });
      }
    }
    return rects;
  }, [grid, gridRes, cell]);

  return (
    <svg width={size} height={size} className="rounded-lg block">
      <rect width={size} height={size} fill="#0b1220" />
      {cells &&
        cells.map((c, i) => (
          <rect
            key={i}
            x={c.x}
            y={c.y}
            width={cell + 0.5}
            height={cell + 0.5}
            fill={c.fill}
          />
        ))}
      {/* axes */}
      <line
        x1={size / 2}
        y1={0}
        x2={size / 2}
        y2={size}
        stroke="rgba(148,163,184,0.2)"
        strokeWidth={1}
      />
      <line
        x1={0}
        y1={size / 2}
        x2={size}
        y2={size / 2}
        stroke="rgba(148,163,184,0.2)"
        strokeWidth={1}
      />
      {data.map((p, i) => {
        const cx = ((p.x + 1) / 2) * size;
        const cy = ((1 - (p.y + 1) / 2)) * size;
        const fill = p.label === 0 ? "#38bdf8" : "#4ade80";
        return (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={3.2}
            fill={fill}
            stroke="#0f172a"
            strokeWidth={1}
          />
        );
      })}
    </svg>
  );
}

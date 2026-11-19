// SVG heatmaps for attention matrices and activations (values come straight from the model).

export function AttentionMap({ matrix, tokens, highlightRow, size = 220, color = "61,165,255", title }) {
  const n = matrix.length;
  const cell = size / n;
  return (
    <div>
      {title && <div className="text-[11px] text-muted font-mono mb-1">{title}</div>}
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="rounded bg-[#080b10] border border-line max-w-full h-auto">
        {matrix.map((row, i) =>
          row.map((v, j) =>
            j <= i ? (
              <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell} height={cell} fill={`rgba(${color},${Math.min(1, v * 1.15)})`}>
                <title>{`query ${i} «${tokens[i]}» → key ${j} «${tokens[j]}»: ${v.toFixed(3)}`}</title>
              </rect>
            ) : null,
          ),
        )}
        {highlightRow != null && highlightRow < n && (
          <rect x={0} y={highlightRow * cell} width={size} height={cell} fill="none" stroke="#fff" strokeWidth={1} />
        )}
      </svg>
    </div>
  );
}

export function ActivationMap({ matrix, tokens, highlightRow, width = 560 }) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const cw = width / cols;
  const ch = Math.max(8, Math.min(16, 260 / rows));
  const max = Math.max(1e-6, ...matrix.flat().map((v) => Math.abs(v)));
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg width={width + 34} height={rows * ch} className="block">
        {matrix.map((row, i) => (
          <g key={i}>
            <text x={0} y={i * ch + ch * 0.75} fontSize={Math.min(11, ch)} fill={i === highlightRow ? "#fff" : "#7d8a9c"} fontFamily="monospace">
              {tokens[i]?.startsWith("<") ? "◆" : tokens[i]}
            </text>
            {row.map((v, j) => {
              const a = Math.min(1, Math.abs(v) / max);
              return (
                <rect key={j} x={34 + j * cw} y={i * ch} width={cw} height={ch - 1}
                  fill={v >= 0 ? `rgba(255,61,90,${a})` : `rgba(61,165,255,${a})`}>
                  <title>{`pos ${i} «${tokens[i]}», dim ${j}: ${v}`}</title>
                </rect>
              );
            })}
            {i === highlightRow && <rect x={34} y={i * ch} width={width} height={ch - 1} fill="none" stroke="#fff" />}
          </g>
        ))}
      </svg>
      <div className="text-[11px] text-muted mt-1">
        Filas = posiciones, columnas = dimensiones del flujo residual. Rojo = positivo, azul = negativo (escala ±{max.toFixed(2)}).
      </div>
    </div>
  );
}

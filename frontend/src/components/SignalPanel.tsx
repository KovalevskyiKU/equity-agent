import type { Signals } from '../api'

const pct = (x: number | null) => (x == null ? '—' : (x * 100).toFixed(1) + '%')

export function SignalPanel({ s }: { s: Signals | null }) {
  if (!s) return <div className="signals muted">select an instrument</div>
  return (
    <div className="signals">
      <span className="sig">
        <b>{s.symbol}</b> <span className="muted">{s.asset_class}</span>
      </span>
      <span className="sig">
        last <b>{s.last_price.toLocaleString()}</b>
      </span>
      <span className={'sig trend-' + s.trend}>trend {s.trend === 'up' ? '▲' : '▼'}</span>
      <span className="sig">vol {pct(s.ann_vol)}</span>
      <span className="sig">3m mom {pct(s.momentum_3m)}</span>
    </div>
  )
}

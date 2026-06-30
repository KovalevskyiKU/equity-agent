import type { Portfolio } from '../api'

export function PortfolioView({ p }: { p: Portfolio | null }) {
  if (!p) return <div className="panel muted">portfolio loading…</div>
  const m = p.monitor || {}
  const equity = (m.equity as number) ?? p.cash
  const tr = m.total_return
  return (
    <div className="panel">
      <div className="panel-h">Portfolio</div>
      <div className="kpis">
        <div>
          <span>Equity</span>
          <b>${Math.round(equity).toLocaleString()}</b>
        </div>
        <div>
          <span>Cash</span>
          <b>${Math.round(p.cash).toLocaleString()}</b>
        </div>
        <div>
          <span>Total ret</span>
          <b>{tr != null ? (tr * 100).toFixed(2) + '%' : '—'}</b>
        </div>
      </div>
      {p.positions.length === 0 ? (
        <div className="muted small">no open positions</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Sym</th>
              <th>Qty</th>
              <th>Avg</th>
              <th>Last</th>
              <th>uPnL</th>
            </tr>
          </thead>
          <tbody>
            {p.positions.map((x) => (
              <tr key={x.symbol}>
                <td>{x.symbol}</td>
                <td>{x.qty}</td>
                <td>{x.avg_price.toFixed(2)}</td>
                <td>{x.last.toFixed(2)}</td>
                <td className={x.unrealized_pnl >= 0 ? 'pos' : 'neg'}>
                  {x.unrealized_pnl.toFixed(0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

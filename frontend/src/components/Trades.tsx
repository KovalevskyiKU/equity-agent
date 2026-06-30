import type { Trade } from '../api'

export function Trades({ trades }: { trades: Trade[] }) {
  return (
    <div className="panel">
      <div className="panel-h">Recent trades</div>
      {trades.length === 0 ? (
        <div className="muted small">no trades yet</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Time</th>
              <th>Sym</th>
              <th>Side</th>
              <th>Qty</th>
              <th>Px</th>
            </tr>
          </thead>
          <tbody>
            {trades.slice(0, 12).map((t, i) => (
              <tr key={i}>
                <td className="small">{t.time.slice(0, 16)}</td>
                <td>{t.symbol}</td>
                <td className={t.side === 'BUY' ? 'pos' : 'neg'}>{t.side}</td>
                <td>{t.qty}</td>
                <td>{t.price.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

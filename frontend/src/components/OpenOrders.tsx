import { useEffect, useState } from 'react'
import type { OpenOrder } from '../api'
import { api } from '../api'

export function OpenOrders({
  orders,
  onDone,
}: {
  orders: OpenOrder[] | null
  onDone: () => void
}) {
  // Prefer the live list pushed via WS (portfolio.open_orders). Fall back to a
  // one-shot REST fetch when the portfolio hasn't supplied the field yet.
  const [fallback, setFallback] = useState<OpenOrder[]>([])
  const [cancelling, setCancelling] = useState<number | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    if (orders !== null) return
    api.getOpenOrders().then(setFallback).catch(() => setFallback([]))
  }, [orders])

  const list = orders ?? fallback

  const cancel = async (id: number) => {
    setErr('')
    setCancelling(id)
    try {
      await api.cancelOrder(id)
      // Drop it locally for snappy feedback; WS + refresh reconcile shortly.
      if (orders === null) setFallback((xs) => xs.filter((o) => o.id !== id))
      onDone()
    } catch (e: unknown) {
      setErr('⚠ ' + (e as Error).message)
    } finally {
      setCancelling(null)
    }
  }

  if (list.length === 0) return null

  return (
    <div className="panel">
      <div className="panel-h">Open orders</div>
      <table className="tbl">
        <thead>
          <tr>
            <th>Sym</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Limit</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {list.map((o) => (
            <tr key={o.id}>
              <td>{o.symbol}</td>
              <td className={o.side === 'BUY' ? 'pos' : 'neg'}>{o.side}</td>
              <td>{o.qty}</td>
              <td>{o.limit_price.toFixed(2)}</td>
              <td>
                <button
                  className="order-cancel"
                  disabled={cancelling === o.id}
                  onClick={() => cancel(o.id)}
                >
                  {cancelling === o.id ? '…' : 'Cancel'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {err && <div className="ticket-msg small err">{err}</div>}
    </div>
  )
}

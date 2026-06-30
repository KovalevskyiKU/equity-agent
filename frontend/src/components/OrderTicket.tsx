import { useState } from 'react'
import { api } from '../api'

export function OrderTicket({ symbol, onDone }: { symbol: string; onDone: () => void }) {
  const [qty, setQty] = useState('1')
  const [msg, setMsg] = useState('')

  const submit = async (side: string) => {
    setMsg('…')
    try {
      const r = await api.placeOrder({ symbol, side, qty: parseFloat(qty) })
      setMsg(`✓ ${side} ${r.qty} ${r.symbol} @ ${Number(r.exec_price).toFixed(2)}`)
      onDone()
    } catch (e: unknown) {
      setMsg('⚠ ' + (e as Error).message)
    }
  }

  return (
    <div className="panel ticket">
      <div className="panel-h">
        Order — {symbol} <span className="paper-badge">PAPER</span>
      </div>
      <label className="small muted">Quantity</label>
      <input value={qty} onChange={(e) => setQty(e.target.value)} type="number" min="0" step="any" />
      <div className="ticket-btns">
        <button className="buy" onClick={() => submit('BUY')}>
          Buy
        </button>
        <button className="sell" onClick={() => submit('SELL')}>
          Sell
        </button>
      </div>
      {msg && <div className="ticket-msg small">{msg}</div>}
    </div>
  )
}

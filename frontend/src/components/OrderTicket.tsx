import { useState } from 'react'
import type { OrderPreview, Venue } from '../api'
import { api, isPreview } from '../api'

const VENUES: { value: Venue; label: string }[] = [
  { value: 'paper', label: 'Paper' },
  { value: 'ibkr', label: 'IBKR' },
  { value: 'binance', label: 'Binance' },
]

export function OrderTicket({ symbol, onDone }: { symbol: string; onDone: () => void }) {
  const [qty, setQty] = useState('1')
  const [venue, setVenue] = useState<Venue>('paper')
  const [msg, setMsg] = useState('')
  const [msgKind, setMsgKind] = useState<'ok' | 'err' | 'info'>('info')
  const [pending, setPending] = useState(false)
  // Live-venue preview awaiting explicit confirmation.
  const [preview, setPreview] = useState<OrderPreview | null>(null)

  const live = venue !== 'paper'
  const badge = live ? venue.toUpperCase() : 'PAPER'

  const say = (kind: 'ok' | 'err' | 'info', text: string) => {
    setMsgKind(kind)
    setMsg(text)
  }

  const submit = async (side: string) => {
    const q = parseFloat(qty)
    if (!Number.isFinite(q) || q <= 0) {
      say('err', '⚠ enter a positive quantity')
      return
    }
    setPreview(null)
    setPending(true)
    say('info', '…')
    try {
      // Paper fills immediately; live venues return a preview first (confirm=false).
      const r = await api.placeOrder({ symbol, side, qty: q, venue })
      if (isPreview(r)) {
        setPreview(r)
        say('info', `${r.venue.toUpperCase()} preview — review and confirm`)
      } else if ('exec_price' in r) {
        say('ok', `✓ ${side} ${r.qty} ${r.symbol} @ ${Number(r.exec_price).toFixed(2)}`)
        onDone()
      } else {
        say('ok', `✓ ${r.filled}: ${side} ${r.qty} ${r.symbol} (${r.venue.toUpperCase()})`)
        onDone()
      }
    } catch (e: unknown) {
      say('err', '⚠ ' + (e as Error).message)
    } finally {
      setPending(false)
    }
  }

  const confirmPreview = async () => {
    if (!preview) return
    setPending(true)
    say('info', '…transmitting')
    try {
      const r = await api.placeOrder({
        symbol: preview.symbol,
        side: preview.side,
        qty: preview.qty,
        venue: preview.venue,
        confirm: true,
      })
      if (isPreview(r)) {
        // Shouldn't happen with confirm=true, but stay defensive.
        say('err', '⚠ venue still requires confirmation')
      } else if ('venue' in r) {
        say('ok', `✓ ${r.filled}: ${r.side} ${r.qty} ${r.symbol} (${r.venue.toUpperCase()})`)
        onDone()
      } else {
        say('ok', `✓ ${r.filled} ${r.side} ${r.qty} ${r.symbol}`)
        onDone()
      }
    } catch (e: unknown) {
      say('err', '⚠ ' + (e as Error).message)
    } finally {
      setPreview(null)
      setPending(false)
    }
  }

  const cancelPreview = () => {
    setPreview(null)
    say('info', 'cancelled — nothing was sent')
  }

  return (
    <div className="panel ticket">
      <div className="panel-h">
        Order — {symbol}{' '}
        <span className={'paper-badge' + (live ? ' live-badge' : '')}>{badge}</span>
      </div>

      <label className="small muted">Venue</label>
      <select
        className="venue-select"
        value={venue}
        disabled={pending || !!preview}
        onChange={(e) => {
          setVenue(e.target.value as Venue)
          setPreview(null)
          setMsg('')
        }}
      >
        {VENUES.map((v) => (
          <option key={v.value} value={v.value}>
            {v.label}
          </option>
        ))}
      </select>

      <label className="small muted">Quantity</label>
      <input
        value={qty}
        onChange={(e) => setQty(e.target.value)}
        type="number"
        min="0"
        step="any"
        disabled={!!preview}
      />

      <div className="ticket-btns">
        <button className="buy" disabled={pending || !!preview} onClick={() => submit('BUY')}>
          Buy
        </button>
        <button className="sell" disabled={pending || !!preview} onClick={() => submit('SELL')}>
          Sell
        </button>
      </div>

      {preview && (
        <div className="confirm-box">
          <div className="small">
            <b>{preview.side}</b> {preview.qty} {preview.symbol} on{' '}
            <b>{preview.venue.toUpperCase()}</b>
          </div>
          <div className="small muted">
            est price {Number(preview.est_price).toFixed(2)} · est notional $
            {Number(preview.est_notional).toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </div>
          <div className="confirm-btns">
            <button className="buy" disabled={pending} onClick={confirmPreview}>
              Confirm
            </button>
            <button className="confirm-cancel" disabled={pending} onClick={cancelPreview}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {msg && <div className={'ticket-msg small ' + msgKind}>{msg}</div>}
    </div>
  )
}

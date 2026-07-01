import { useState } from 'react'
import type { OrderPreview, OrderType, Venue } from '../api'
import { api, isPreview, isResting } from '../api'

const VENUES: { value: Venue; label: string }[] = [
  { value: 'paper', label: 'Paper' },
  { value: 'ibkr', label: 'IBKR' },
  { value: 'binance', label: 'Binance' },
]

export function OrderTicket({ symbol, onDone }: { symbol: string; onDone: () => void }) {
  const [qty, setQty] = useState('1')
  const [venue, setVenue] = useState<Venue>('paper')
  const [orderType, setOrderType] = useState<OrderType>('market')
  const [limitPrice, setLimitPrice] = useState('')
  const [msg, setMsg] = useState('')
  const [msgKind, setMsgKind] = useState<'ok' | 'err' | 'info'>('info')
  const [pending, setPending] = useState(false)
  // Live-venue preview awaiting explicit confirmation.
  const [preview, setPreview] = useState<OrderPreview | null>(null)

  const live = venue !== 'paper'
  const isLimit = orderType === 'limit'
  const isStop = orderType === 'stop'
  // Resting limit/stop orders are paper-only; block submit on live venues.
  const priced = isLimit || isStop
  const restingOnLive = priced && live
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
    if (restingOnLive) {
      say('err', `⚠ ${orderType} orders are paper-only`)
      return
    }
    let lp: number | undefined
    if (priced) {
      lp = parseFloat(limitPrice)
      if (!Number.isFinite(lp) || lp <= 0) {
        say('err', `⚠ enter a positive ${isStop ? 'stop' : 'limit'} price`)
        return
      }
    }
    setPreview(null)
    setPending(true)
    say('info', '…')
    try {
      // Paper market fills immediately; paper limit/stop rests; live venues
      // return a preview first (confirm=false).
      const r = await api.placeOrder(
        priced
          ? { symbol, side, qty: q, venue, order_type: orderType, limit_price: lp }
          : { symbol, side, qty: q, venue, order_type: 'market' },
      )
      if (isResting(r)) {
        const label = r.kind === 'stop' ? 'stop' : 'limit'
        say('ok', `✓ ${label} order placed @ ${Number(r.limit_price).toFixed(2)}`)
        onDone()
      } else if (isPreview(r)) {
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
      } else if ('filled' in r) {
        say('ok', `✓ ${r.filled} ${r.side} ${r.qty} ${r.symbol}`)
        onDone()
      } else {
        // A resting limit order can't come back from a confirm=true live order.
        say('ok', `✓ ${r.side} ${r.qty} ${r.symbol}`)
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

      <label className="small muted">Order type</label>
      <div className="ordertype-toggle">
        <button
          type="button"
          className={'ordertype-btn' + (orderType === 'market' ? ' active' : '')}
          disabled={pending || !!preview}
          onClick={() => {
            setOrderType('market')
            setMsg('')
          }}
        >
          Market
        </button>
        <button
          type="button"
          className={'ordertype-btn' + (orderType === 'limit' ? ' active' : '')}
          disabled={pending || !!preview}
          onClick={() => {
            setOrderType('limit')
            setMsg('')
          }}
        >
          Limit
        </button>
        <button
          type="button"
          className={'ordertype-btn' + (orderType === 'stop' ? ' active' : '')}
          disabled={pending || !!preview}
          onClick={() => {
            setOrderType('stop')
            setMsg('')
          }}
        >
          Stop
        </button>
      </div>

      <label className="small muted">Quantity</label>
      <input
        value={qty}
        onChange={(e) => setQty(e.target.value)}
        type="number"
        min="0"
        step="any"
        disabled={!!preview}
      />

      {priced && (
        <>
          <label className="small muted">
            {isStop ? 'Stop (trigger) price' : 'Limit price'}
          </label>
          <input
            value={limitPrice}
            onChange={(e) => setLimitPrice(e.target.value)}
            type="number"
            min="0"
            step="any"
            placeholder="e.g. 420.50"
            disabled={!!preview}
          />
        </>
      )}

      <div className="ticket-btns">
        <button
          className="buy"
          disabled={pending || !!preview || restingOnLive}
          onClick={() => submit('BUY')}
        >
          Buy
        </button>
        <button
          className="sell"
          disabled={pending || !!preview || restingOnLive}
          onClick={() => submit('SELL')}
        >
          Sell
        </button>
      </div>

      {restingOnLive && (
        <div className="ticket-msg small info">{orderType} orders are paper-only</div>
      )}

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

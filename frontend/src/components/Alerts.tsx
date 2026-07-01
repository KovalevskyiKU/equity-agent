import { useEffect, useState } from 'react'
import type { Alert } from '../api'
import { api } from '../api'

const KINDS: { value: Alert['kind']; label: string }[] = [
  { value: 'above', label: 'Above' },
  { value: 'below', label: 'Below' },
  { value: 'trend_up', label: 'Trend up' },
  { value: 'trend_down', label: 'Trend down' },
]

// Alerts with a "level" need a positive number; trend alerts don't.
function needsLevel(kind: Alert['kind']): boolean {
  return kind === 'above' || kind === 'below'
}

// Human label, e.g. "▲ above 420", "▼ below 100", "trend up".
function alertLabel(a: Alert): string {
  switch (a.kind) {
    case 'above':
      return `▲ above ${a.level ?? ''}`.trimEnd()
    case 'below':
      return `▼ below ${a.level ?? ''}`.trimEnd()
    case 'trend_up':
      return 'trend up'
    case 'trend_down':
      return 'trend down'
  }
}

export function Alerts({
  symbol,
  alerts,
  onDone,
}: {
  symbol: string
  alerts: Alert[] | null
  onDone: () => void
}) {
  const [kind, setKind] = useState<Alert['kind']>('above')
  const [level, setLevel] = useState('')
  const [fallback, setFallback] = useState<Alert[]>([])
  const [deleting, setDeleting] = useState<number | null>(null)
  const [pending, setPending] = useState(false)
  const [err, setErr] = useState('')

  // Prefer the live list pushed via WS (portfolio.alerts). Fall back to a
  // one-shot REST fetch when the portfolio hasn't supplied the field yet.
  useEffect(() => {
    if (alerts !== null) return
    api.getAlerts().then(setFallback).catch(() => setFallback([]))
  }, [alerts])

  const list = alerts ?? fallback

  const add = async () => {
    setErr('')
    let lvl: number | undefined
    if (needsLevel(kind)) {
      lvl = parseFloat(level)
      if (!Number.isFinite(lvl) || lvl <= 0) {
        setErr('⚠ enter a positive level')
        return
      }
    }
    setPending(true)
    try {
      await api.createAlert({ symbol, kind, level: lvl })
      setLevel('')
      onDone()
    } catch (e: unknown) {
      setErr('⚠ ' + (e as Error).message)
    } finally {
      setPending(false)
    }
  }

  const remove = async (id: number) => {
    setErr('')
    setDeleting(id)
    try {
      await api.deleteAlert(id)
      // Drop it locally for snappy feedback; WS + refresh reconcile shortly.
      if (alerts === null) setFallback((xs) => xs.filter((a) => a.id !== id))
      onDone()
    } catch (e: unknown) {
      setErr('⚠ ' + (e as Error).message)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="panel">
      <div className="panel-h">Alerts</div>

      <div className="alert-form">
        <select
          className="venue-select"
          value={kind}
          disabled={pending}
          onChange={(e) => setKind(e.target.value as Alert['kind'])}
        >
          {KINDS.map((k) => (
            <option key={k.value} value={k.value}>
              {k.label}
            </option>
          ))}
        </select>
        {needsLevel(kind) && (
          <input
            className="alert-level"
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            type="number"
            min="0"
            step="any"
            placeholder="level"
            disabled={pending}
          />
        )}
        <button className="alert-add" disabled={pending} onClick={add}>
          Add
        </button>
      </div>
      <div className="small muted alert-sym">on {symbol}</div>

      {list.length === 0 ? (
        <div className="muted small">no alerts</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Sym</th>
              <th>Alert</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {list.map((a) => (
              <tr key={a.id} className={a.status === 'triggered' ? 'alert-triggered' : ''}>
                <td>{a.symbol}</td>
                <td className="small">{alertLabel(a)}</td>
                <td className={'small ' + (a.status === 'triggered' ? 'pos' : 'muted')}>
                  {a.status}
                </td>
                <td>
                  <button
                    className="order-cancel"
                    disabled={deleting === a.id}
                    onClick={() => remove(a.id)}
                    aria-label="delete alert"
                  >
                    {deleting === a.id ? '…' : '×'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {err && <div className="ticket-msg small err">{err}</div>}
    </div>
  )
}

// Typed client for the equity-agent FastAPI backend.

export interface Instrument {
  symbol: string
  asset_class: 'equity' | 'crypto'
  role: string
}
export interface Bar {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}
export interface Signals {
  symbol: string
  asset_class: string
  last_price: number
  sma_fast: number | null
  sma_slow: number | null
  trend: 'up' | 'down'
  ann_vol: number | null
  momentum_3m: number | null
}
export interface Position {
  symbol: string
  qty: number
  avg_price: number
  last: number
  market_value: number
  unrealized_pnl: number
}
export interface OpenOrder {
  id: number
  symbol: string
  side: string
  qty: number
  limit_price: number
  created_at: string
}
export interface Portfolio {
  has_account: boolean
  cash: number
  starting_cash: number
  positions: Position[]
  monitor: Record<string, number>
  equity_curve: { time: string; equity: number }[]
  open_orders?: OpenOrder[]
}
export interface Trade {
  time: string
  symbol: string
  side: string
  qty: number
  price: number
  pnl: number | null
}
export type Venue = 'paper' | 'ibkr' | 'binance'

// Paper venue fill (immediate).
export interface OrderResult {
  filled: string
  symbol: string
  side: string
  qty: number
  exec_price: number
  fee: number
  cash: number
  equity: number
}

// ibkr/binance with confirm=false: preview only, nothing sent.
export interface OrderPreview {
  preview: true
  requires_confirmation: true
  venue: Venue
  symbol: string
  side: string
  qty: number
  est_price: number
  est_notional: number
}

// ibkr/binance with confirm=true: transmitted to the live venue.
export interface OrderTransmitted {
  filled: string
  venue: Venue
  symbol: string
  side: string
  qty: number
}

// order_type:"limit" (paper only): a resting limit order was created.
export interface OrderResting {
  order_id: number
  status: 'open'
  symbol: string
  side: string
  qty: number
  limit_price: number
}

export type OrderType = 'market' | 'limit'

export interface OrderBody {
  symbol: string
  side: string
  qty: number
  venue?: Venue
  price?: number
  confirm?: boolean
  order_type?: OrderType
  limit_price?: number
}

export type OrderResponse = OrderResult | OrderPreview | OrderTransmitted | OrderResting

export function isPreview(r: OrderResponse): r is OrderPreview {
  return (r as OrderPreview).preview === true
}

export function isResting(r: OrderResponse): r is OrderResting {
  return (r as OrderResting).status === 'open'
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url} -> ${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  instruments: () => get<Instrument[]>('/api/instruments'),
  bars: (sym: string, limit = 750) => get<Bar[]>(`/api/bars/${sym}?limit=${limit}`),
  signals: (sym: string) => get<Signals>(`/api/signals/${sym}`),
  portfolio: () => get<Portfolio>('/api/portfolio'),
  trades: () => get<Trade[]>('/api/trades'),
  placeOrder: async (body: OrderBody): Promise<OrderResponse> => {
    const r = await fetch('/api/orders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await r.json()
    if (!r.ok) throw new Error(data.detail ?? `order failed (${r.status})`)
    return data as OrderResponse
  },
  getOpenOrders: () => get<OpenOrder[]>('/api/orders/open'),
  cancelOrder: async (id: number): Promise<{ id: number; status: string }> => {
    const r = await fetch(`/api/orders/${id}`, { method: 'DELETE' })
    const data = await r.json()
    if (!r.ok) throw new Error(data.detail ?? `cancel failed (${r.status})`)
    return data as { id: number; status: string }
  },
}

// Live portfolio updates over the /ws/portfolio WebSocket. Pushes the same
// shape as GET /api/portfolio (~every 3s). Auto-reconnects on close/error with
// a small backoff. Returns a cleanup function that tears down the socket and
// cancels any pending reconnect (no leaked sockets on unmount).
export function connectPortfolioWS(
  onMessage: (p: Portfolio) => void,
  backoffMs = 2000,
): () => void {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${location.host}/ws/portfolio`
  let ws: WebSocket | null = null
  let timer: ReturnType<typeof setTimeout> | null = null
  let closed = false

  const connect = () => {
    if (closed) return
    ws = new WebSocket(url)
    ws.onmessage = (ev) => {
      try {
        onMessage(JSON.parse(ev.data) as Portfolio)
      } catch {
        // ignore malformed frames
      }
    }
    const scheduleReconnect = () => {
      if (closed || timer) return
      timer = setTimeout(() => {
        timer = null
        connect()
      }, backoffMs)
    }
    ws.onclose = scheduleReconnect
    ws.onerror = () => {
      // onerror is typically followed by onclose, but close defensively so a
      // socket stuck in a failed state still triggers a reconnect.
      try {
        ws?.close()
      } catch {
        // already closing/closed
      }
      scheduleReconnect()
    }
  }

  connect()

  return () => {
    closed = true
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
    if (ws) {
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      ws.close()
      ws = null
    }
  }
}

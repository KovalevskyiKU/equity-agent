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
export interface Portfolio {
  has_account: boolean
  cash: number
  starting_cash: number
  positions: Position[]
  monitor: Record<string, number>
  equity_curve: { time: string; equity: number }[]
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

export interface OrderBody {
  symbol: string
  side: string
  qty: number
  venue?: Venue
  price?: number
  confirm?: boolean
}

export type OrderResponse = OrderResult | OrderPreview | OrderTransmitted

export function isPreview(r: OrderResponse): r is OrderPreview {
  return (r as OrderPreview).preview === true
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
}

// Live portfolio updates over the /ws/portfolio WebSocket. Pushes the same
// shape as GET /api/portfolio (~every 3s). Returns the socket for cleanup.
export function connectPortfolioWS(onMessage: (p: Portfolio) => void): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws/portfolio`)
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data) as Portfolio)
    } catch {
      // ignore malformed frames
    }
  }
  return ws
}

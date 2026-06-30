import { useCallback, useEffect, useState } from 'react'
import './App.css'
import type { Bar, Instrument, Portfolio, Signals, Trade } from './api'
import { api } from './api'
import { Chart } from './components/Chart'
import { OrderTicket } from './components/OrderTicket'
import { PortfolioView } from './components/Portfolio'
import { SignalPanel } from './components/SignalPanel'
import { Trades } from './components/Trades'
import { Watchlist } from './components/Watchlist'

export default function App() {
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [symbol, setSymbol] = useState('SPY')
  const [bars, setBars] = useState<Bar[]>([])
  const [signals, setSignals] = useState<Signals | null>(null)
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])

  useEffect(() => {
    api.instruments().then(setInstruments).catch(() => {})
  }, [])

  useEffect(() => {
    api.bars(symbol).then(setBars).catch(() => setBars([]))
    api.signals(symbol).then(setSignals).catch(() => setSignals(null))
  }, [symbol])

  const refresh = useCallback(() => {
    api.portfolio().then(setPortfolio).catch(() => {})
    api.trades().then(setTrades).catch(() => {})
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh])

  return (
    <div className="app">
      <header className="topbar">
        <b>equity-agent</b>
        <span className="muted">trading cockpit</span>
        <span className="paper-badge">PAPER</span>
      </header>
      <div className="grid">
        <aside className="left">
          <Watchlist items={instruments} selected={symbol} onSelect={setSymbol} />
        </aside>
        <main className="center">
          <SignalPanel s={signals} />
          <div className="chart">
            {bars.length ? (
              <Chart bars={bars} />
            ) : (
              <div className="muted center-msg">
                No data for {symbol} — run <code>eqa ingest</code> / <code>ingest-crypto</code>.
              </div>
            )}
          </div>
        </main>
        <aside className="right">
          <OrderTicket symbol={symbol} onDone={refresh} />
          <PortfolioView p={portfolio} />
          <Trades trades={trades} />
        </aside>
      </div>
    </div>
  )
}

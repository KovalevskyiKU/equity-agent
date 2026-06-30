import { CandlestickSeries, ColorType, LineSeries, createChart } from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import type { Bar } from '../api'

function sma(bars: Bar[], n: number) {
  const out: { time: string; value: number }[] = []
  for (let i = n - 1; i < bars.length; i++) {
    let s = 0
    for (let j = i - n + 1; j <= i; j++) s += bars[j].close
    out.push({ time: bars[i].time, value: s / n })
  }
  return out
}

export function Chart({ bars }: { bars: Bar[] }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || bars.length === 0) return
    const chart = createChart(el, {
      width: el.clientWidth || 600,
      height: el.clientHeight || 400,
      layout: { background: { type: ColorType.Solid, color: '#0e1117' }, textColor: '#c9d1d9' },
      grid: { vertLines: { color: '#1b2230' }, horzLines: { color: '#1b2230' } },
      rightPriceScale: { borderColor: '#1b2230' },
      timeScale: { borderColor: '#1b2230' },
    })
    const onResize = () => chart.resize(el.clientWidth, el.clientHeight)
    window.addEventListener('resize', onResize)
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    })
    candle.setData(
      bars.map((b) => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close })),
    )
    if (bars.length > 20) chart.addSeries(LineSeries, { color: '#e3b341', lineWidth: 1 }).setData(sma(bars, 20))
    if (bars.length > 100) chart.addSeries(LineSeries, { color: '#a371f7', lineWidth: 1 }).setData(sma(bars, 100))
    chart.timeScale().fitContent()
    return () => {
      window.removeEventListener('resize', onResize)
      chart.remove()
    }
  }, [bars])
  return <div ref={ref} style={{ width: '100%', height: '100%' }} />
}

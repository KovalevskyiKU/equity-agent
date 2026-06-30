import { ColorType, LineSeries, createChart } from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import type { Portfolio } from '../api'

export function EquityCurve({ data }: { data: Portfolio['equity_curve'] }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el || !data || data.length === 0) return
    const chart = createChart(el, {
      width: el.clientWidth || 300,
      height: el.clientHeight || 160,
      layout: { background: { type: ColorType.Solid, color: '#0e1117' }, textColor: '#c9d1d9' },
      grid: { vertLines: { color: '#1b2230' }, horzLines: { color: '#1b2230' } },
      rightPriceScale: { borderColor: '#1b2230' },
      timeScale: { borderColor: '#1b2230' },
    })
    const onResize = () => chart.resize(el.clientWidth, el.clientHeight)
    window.addEventListener('resize', onResize)
    const line = chart.addSeries(LineSeries, { color: '#58a6ff', lineWidth: 2 })
    line.setData(data.map((d) => ({ time: d.time, value: d.equity })))
    chart.timeScale().fitContent()
    return () => {
      window.removeEventListener('resize', onResize)
      chart.remove()
    }
  }, [data])

  return (
    <div className="panel">
      <div className="panel-h">Equity curve</div>
      {!data || data.length === 0 ? (
        <div className="muted small">no equity history yet</div>
      ) : (
        <div ref={ref} className="equity-chart" />
      )}
    </div>
  )
}

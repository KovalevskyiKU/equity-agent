import type { Instrument } from '../api'

export function Watchlist({
  items,
  selected,
  onSelect,
}: {
  items: Instrument[]
  selected: string
  onSelect: (s: string) => void
}) {
  const eq = items.filter((i) => i.asset_class === 'equity')
  const cr = items.filter((i) => i.asset_class === 'crypto')
  const group = (title: string, list: Instrument[]) => (
    <div>
      <div className="wl-group">{title}</div>
      {list.map((i) => (
        <div
          key={i.symbol}
          className={'wl-item' + (i.symbol === selected ? ' active' : '')}
          onClick={() => onSelect(i.symbol)}
        >
          {i.symbol}
          {i.role === 'benchmark' ? ' ★' : ''}
        </div>
      ))}
    </div>
  )
  return (
    <div className="watchlist">
      {group(`Equities (${eq.length})`, eq)}
      {group(`Crypto (${cr.length})`, cr)}
    </div>
  )
}

import React, { useMemo, useState } from "react";

export default function EmergencyTradeModal({ trades, onSubmit, onClose }) {
  const [selected, setSelected] = useState({});
  const tradeRows = useMemo(() => trades || [], [trades]);

  function toggle(id) {
    setSelected((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  function submit() {
    const tradeIds = Object.keys(selected)
      .filter((k) => selected[k])
      .map((k) => Number(k));
    onSubmit(tradeIds);
  }

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", display: "grid", placeItems: "center" }}>
      <div className="card" style={{ width: "90%", maxHeight: "80vh", overflow: "auto" }}>
        <h3>Emergency Trade Review</h3>
        <p>Check trades that were executed due to financial emergency.</p>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th>ID</th><th>Timestamp</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Emergency</th>
            </tr>
          </thead>
          <tbody>
            {tradeRows.map((t, i) => (
              <tr key={i}>
                <td>{t.id ?? i + 1}</td>
                <td>{String(t.timestamp)}</td>
                <td>{t.symbol}</td>
                <td>{t.buy_sell}</td>
                <td>{t.quantity}</td>
                <td>{t.price}</td>
                <td>
                  <input type="checkbox" checked={!!selected[t.id ?? i + 1]} onChange={() => toggle(t.id ?? i + 1)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button onClick={submit}>Submit Flags</button>
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

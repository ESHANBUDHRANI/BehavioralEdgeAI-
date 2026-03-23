import React, { useEffect, useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

/**
 * FIX: original code used file:/// URLs inside iframes, which browsers block
 * due to same-origin security policy when the page is served over HTTP.
 *
 * New approach:
 *  - Backend /api/charts/{sessionId} already returns absolute file paths.
 *  - We extract the filename from each path and load the chart HTML via a
 *    new backend endpoint /api/chart-file/{sessionId}/{filename} that reads
 *    the file and serves it as text/html.
 *  - If you prefer, you can also embed the chart content inline using a
 *    blob URL — see the blob approach below.
 */
export default function ChartsGrid({ sessionId, charts }) {
  const [chartContents, setChartContents] = useState({});

  useEffect(() => {
    if (!charts || charts.length === 0) return;

    charts.forEach(async (chartPath) => {
      // Extract just the filename from the absolute path
      const filename = chartPath.replace(/\\/g, "/").split("/").pop();
      try {
        const { data } = await axios.get(
          `${API}/api/chart-file/${sessionId}/${filename}`,
          { responseType: "text" }
        );
        // Create a blob URL so the iframe loads real HTML over HTTP
        const blob = new Blob([data], { type: "text/html" });
        const url = URL.createObjectURL(blob);
        setChartContents((prev) => ({ ...prev, [filename]: url }));
      } catch {
        // Silently skip charts that fail to load
      }
    });

    // Revoke blob URLs on cleanup
    return () => {
      Object.values(chartContents).forEach((url) => URL.revokeObjectURL(url));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [charts, sessionId]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        gap: 12,
      }}
    >
      {(charts || []).map((chartPath, idx) => {
        const filename = chartPath.replace(/\\/g, "/").split("/").pop();
        const blobUrl = chartContents[filename];
        return (
          <div className="card" key={idx}>
            <div style={{ marginBottom: 8 }}>Chart {idx + 1}</div>
            {blobUrl ? (
              <iframe
                src={blobUrl}
                title={`chart-${idx}`}
                style={{ width: "100%", height: 360, border: "1px solid #00FF41" }}
                loading="lazy"
              />
            ) : (
              <div
                style={{
                  height: 360,
                  display: "grid",
                  placeItems: "center",
                  border: "1px solid #00FF41",
                  color: "#555",
                }}
              >
                Loading…
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

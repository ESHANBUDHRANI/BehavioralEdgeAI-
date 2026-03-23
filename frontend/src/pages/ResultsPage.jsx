import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ChartsGrid from "../components/ChartsGrid";
import ChatPanel from "../components/ChatPanel";
import { fetchAnalysis, fetchCharts, fetchReport } from "../hooks/useSessionApi";

export default function ResultsPage() {
  const { sessionId } = useParams();
  const [analysis, setAnalysis] = useState(null);
  const [report, setReport] = useState(null);
  const [charts, setCharts] = useState([]);

  useEffect(() => {
    fetchAnalysis(sessionId).then(setAnalysis);
    fetchReport(sessionId).then(setReport).catch(() => null);
    // FIX: ChartsGrid now receives both sessionId and charts array
    fetchCharts(sessionId).then((r) => setCharts(r.charts || []));
  }, [sessionId]);

  const rows = analysis?.model_results?.clustering?.gmm_labels || [];

  return (
    <div className="container" style={{ display: "grid", gap: 16 }}>
      <section className="card">
        <h2>Trades Table</h2>
        <div>Total analyzed clusters rows: {rows.length}</div>
      </section>

      <section className="card">
        <h2>Market Context Summary</h2>
        <pre style={{ whiteSpace: "pre-wrap" }}>
          {JSON.stringify(analysis?.report?.report?.risk_profile || {}, null, 2)}
        </pre>
      </section>

      <section>
        <h2>Visualizations</h2>
        {/* FIX: pass sessionId so ChartsGrid can fetch chart content via HTTP */}
        <ChartsGrid sessionId={sessionId} charts={charts} />
      </section>

      <section className="card">
        <h2>Model Effectiveness Panel</h2>
        <pre style={{ whiteSpace: "pre-wrap" }}>
          {JSON.stringify(
            {
              silhouette_score: analysis?.model_results?.clustering?.confidence,
              anomaly_rate: analysis?.model_results?.anomaly?.confidence,
              explained_variance: analysis?.model_results?.tft_model?.confidence,
            },
            null,
            2
          )}
        </pre>
      </section>

      <section className="card">
        <h2>Behavioral Report</h2>
        <pre style={{ whiteSpace: "pre-wrap" }}>{report?.report_text || "Report loading..."}</pre>
      </section>

      <section>
        <ChatPanel sessionId={sessionId} />
      </section>
    </div>
  );
}

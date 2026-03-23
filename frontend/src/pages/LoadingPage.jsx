import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { subscribeProgress } from "../hooks/useSessionApi";

export default function LoadingPage() {
  const { sessionId } = useParams();
  const [progress, setProgress] = useState({ stage: "Starting", progress: 0 });
  const navigate = useNavigate();

  useEffect(() => {
    const unsub = subscribeProgress(sessionId, (msg) => {
      setProgress(msg);
      if ((msg.progress || 0) >= 100) {
        navigate(`/results/${sessionId}`);
      }
    });
    return unsub;
  }, [sessionId, navigate]);

  return (
    <div className="container">
      <h2>Processing analysis...</h2>
      <div className="card">
        <div style={{ marginBottom: 8 }}>{progress.stage}</div>
        <div style={{ height: 14, background: "#111", border: "1px solid #00FF41", borderRadius: 999 }}>
          <div style={{ width: `${progress.progress || 0}%`, height: "100%", background: "#00FF41", transition: "all 300ms ease-in-out" }} />
        </div>
        <div style={{ marginTop: 8 }}>{progress.progress || 0}%</div>
      </div>
    </div>
  );
}

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import EmergencyTradeModal from "../components/EmergencyTradeModal";
import { sendEmergencyFlags, uploadTradeFile } from "../hooks/useSessionApi";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [sessionId, setSessionId] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const data = await uploadTradeFile(file);
      setPreview(data.trade_preview || []);
      setSessionId(data.session_id);
    } catch (err) {
      setError(err?.response?.data?.detail || "Upload failed. Please check your file format.");
    } finally {
      setLoading(false);
    }
  }

  async function submitEmergency(tradeIds) {
    await sendEmergencyFlags(sessionId, tradeIds);
    navigate(`/loading/${sessionId}`);
  }

  return (
    <div className="container">
      <h1>Behavioral Trading Analysis</h1>
      <motion.div
        className="card"
        whileHover={{ scale: 1.02 }}
        transition={{ duration: 0.3 }}
        style={{ borderStyle: "dashed", minHeight: 200, display: "grid", placeItems: "center" }}
      >
        <div>
          {/* FIX: added .csv to accepted file types — CSV is the primary input format */}
          <input
            type="file"
            accept=".csv,.pdf,.png,.jpg,.jpeg,.webp"
            onChange={(e) => {
              setFile(e.target.files?.[0]);
              setError("");
            }}
          />
          <button onClick={handleUpload} disabled={!file || loading} style={{ marginLeft: 8 }}>
            {loading ? "Uploading..." : "Upload"}
          </button>
          <div style={{ marginTop: 8, fontSize: 12, color: "#888" }}>
            Accepted formats: CSV, PDF, PNG, JPG, WEBP
          </div>
        </div>
      </motion.div>
      {error && (
        <div className="card" style={{ borderColor: "#ff4444", color: "#ff4444", marginTop: 8 }}>
          {error}
        </div>
      )}
      {preview && (
        <EmergencyTradeModal
          trades={preview}
          onSubmit={submitEmergency}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
  );
}

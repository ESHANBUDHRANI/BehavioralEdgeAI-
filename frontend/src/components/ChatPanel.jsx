import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { computeCounterfactual, fetchChatHistory, sendChat } from "../hooks/useSessionApi";

const QUICK_PROMPTS = [
  "What are my worst biases?",
  "Which stocks match my style?",
  "What if I sized down after losses?",
  "How would I survive a crash?",
  "Explain my most unusual trade",
  "What is my risk profile?"
];

export default function ChatPanel({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [scenario, setScenario] = useState({ variable: "position_size", multiplier: 0.5 });
  const [counterfactual, setCounterfactual] = useState(null);

  useEffect(() => {
    fetchChatHistory(sessionId).then((res) => setMessages(res.history || []));
  }, [sessionId]);

  async function submitMessage(value) {
    const message = value ?? text;
    if (!message.trim()) return;
    setMessages((prev) => [...prev, { role: "user", message }]);
    setText("");
    const reply = await sendChat(sessionId, message);
    setMessages((prev) => [...prev, { role: "assistant", message: reply.response, intent: reply.intent, chunks_used: JSON.stringify(reply.sources || []) }]);
  }

  async function runCounterfactual() {
    const res = await computeCounterfactual(sessionId, { ...scenario, original_pnl: 1000 });
    setCounterfactual(res.result);
  }

  return (
    <div className="card">
      <h3>LLM Chatbot</h3>
      <div style={{ maxHeight: 320, overflow: "auto", marginBottom: 12 }}>
        {messages.map((m, i) => (
          <motion.div key={i} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }}>
            <div style={{ textAlign: m.role === "user" ? "right" : "left", marginBottom: 8 }}>
              <span className="card" style={{ display: "inline-block", color: m.role === "user" ? "#000" : "#fff", background: m.role === "user" ? "#00FF41" : "#0A0A0A" }}>
                {m.message}
              </span>
              {m.role === "assistant" && (
                <div style={{ fontSize: 12, color: "#fff", marginTop: 4 }}>
                  intent: {m.intent || "general_explanation"} | sources: {m.chunks_used || "[]"}
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input value={text} onChange={(e) => setText(e.target.value)} style={{ flex: 1 }} onKeyDown={(e) => e.key === "Enter" && submitMessage()} />
        <button onClick={() => submitMessage()}>Send</button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2,minmax(120px,1fr))", gap: 8, marginBottom: 12 }}>
        {QUICK_PROMPTS.map((p) => (
          <button key={p} onClick={() => submitMessage(p)}>{p}</button>
        ))}
      </div>
      <div className="card">
        <h4>Counterfactual Builder</h4>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select value={scenario.variable} onChange={(e) => setScenario((s) => ({ ...s, variable: e.target.value }))}>
            <option value="position_size">Position Size</option>
            <option value="holding_duration">Holding Duration</option>
            <option value="frequency">Trade Frequency</option>
          </select>
          <input type="range" min="0.1" max="2.0" step="0.1" value={scenario.multiplier} onChange={(e) => setScenario((s) => ({ ...s, multiplier: Number(e.target.value) }))} />
          <span>{scenario.multiplier}x</span>
          <button onClick={runCounterfactual}>Compute</button>
        </div>
        {counterfactual && <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(counterfactual, null, 2)}</pre>}
      </div>
    </div>
  );
}

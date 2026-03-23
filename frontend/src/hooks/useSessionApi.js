import axios from "axios";

const API = "http://localhost:8000";

export async function uploadTradeFile(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axios.post(`${API}/api/upload`, form);
  return data;
}

export async function sendEmergencyFlags(sessionId, tradeIds) {
  const { data } = await axios.post(`${API}/api/emergency/${sessionId}`, {
    trade_ids: tradeIds,
    reason: "financial_emergency"
  });
  return data;
}

export function subscribeProgress(sessionId, onMessage) {
  const source = new EventSource(`${API}/api/progress/${sessionId}`);
  source.onmessage = (event) => {
    onMessage(JSON.parse(event.data));
  };
  return () => source.close();
}

export async function fetchAnalysis(sessionId) {
  const { data } = await axios.get(`${API}/api/analysis/${sessionId}`);
  return data;
}

export async function fetchReport(sessionId) {
  const { data } = await axios.get(`${API}/api/report/${sessionId}`);
  return data;
}

export async function fetchCharts(sessionId) {
  const { data } = await axios.get(`${API}/api/charts/${sessionId}`);
  return data;
}

export async function sendChat(sessionId, message) {
  const { data } = await axios.post(`${API}/api/chat/${sessionId}`, { message });
  return data;
}

export async function computeCounterfactual(sessionId, scenario) {
  const { data } = await axios.post(`${API}/api/counterfactual/${sessionId}`, scenario);
  return data;
}

export async function fetchChatHistory(sessionId) {
  const { data } = await axios.get(`${API}/api/chat/history/${sessionId}`);
  return data;
}

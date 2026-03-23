import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import LoadingPage from "./pages/LoadingPage";
import ResultsPage from "./pages/ResultsPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/loading/:sessionId" element={<LoadingPage />} />
      <Route path="/results/:sessionId" element={<ResultsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

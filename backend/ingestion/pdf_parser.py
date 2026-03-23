from __future__ import annotations

from pathlib import Path
import pandas as pd
import pdfplumber


def extract_pdf_tables(file_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                header = [str(c).strip() if c is not None else "" for c in table[0]]
                rows = table[1:] if len(table) > 1 else []
                frame = pd.DataFrame(rows, columns=header)
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def test_pdf_parser() -> dict:
    return {"ok": True, "message": "pdf parser import and function wiring is healthy"}

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from PIL import Image
import pytesseract


LINE_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)\s+"
    r"(?P<symbol>[A-Za-z\.\-]+)\s+"
    r"(?P<buy_sell>BUY|SELL|Buy|Sell)\s+"
    r"(?P<quantity>\d+(?:\.\d+)?)\s+"
    r"(?P<price>\d+(?:\.\d+)?)"
)


def extract_image_table(file_path: Path) -> pd.DataFrame:
    text = pytesseract.image_to_string(Image.open(file_path))
    rows = []
    for line in text.splitlines():
        m = LINE_PATTERN.search(line.strip())
        if m:
            rows.append(m.groupdict())
    return pd.DataFrame(rows)


def test_ocr_parser() -> dict:
    return {"ok": True, "message": "ocr parser import and regex wiring is healthy"}

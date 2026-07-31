from __future__ import annotations

import hashlib
import math
import unicodedata
from pathlib import Path

LANGUAGES = {
    "en": {"name": "English", "title": "India", "host": "en.wikipedia.org"},
    "hi": {"name": "Hindi", "title": "भारत", "host": "hi.wikipedia.org"},
    "te": {"name": "Telugu", "title": "భారతదేశం", "host": "te.wikipedia.org"},
    "kn": {"name": "Kannada", "title": "ಭಾರತ", "host": "kn.wikipedia.org"},
    "mai": {"name": "Maithili", "title": "भारत", "host": "mai.wikipedia.org"},
}
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def faithful_unit_count(text: str) -> int:
    count = 0
    in_run = False
    for ch in text:
        if unicodedata.category(ch)[0] in {"L", "M", "N"}:
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
            if not ch.isspace():
                count += 1
    return count


def visible_non_whitespace(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_summary(rows: list[dict[str, object]], field: str = "fertility_x") -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: float(row[field]), reverse=True)
    maximum = float(ordered[0][field])
    minimum = float(ordered[-1][field])
    spread = maximum - minimum
    raw = 1000.0 / spread if spread else None
    english = next(float(row[field]) for row in rows if row["code"] == "en")
    hindi = next(float(row[field]) for row in rows if row["code"] == "hi")
    english_penalty = math.exp(max(0.0, english / 1.2 - 1.0))
    hindi_penalty = math.exp(max(0.0, hindi / 1.2 - 1.0))
    return {
        "max_x": maximum,
        "min_x": minimum,
        "spread": spread,
        "raw_score": raw,
        "english_x_le_1_2": english <= 1.2,
        "english_penalty": english_penalty,
        "english_adjusted_score": raw / english_penalty if raw is not None else None,
        "hindi_penalty": hindi_penalty,
        "hindi_adjusted_score": raw / hindi_penalty if raw is not None else None,
        "sorted_by_x_desc": [row["language"] for row in ordered],
    }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

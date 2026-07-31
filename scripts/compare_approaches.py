#!/usr/bin/env python3
"""Create a compact cross-approach report without conflating corpus/metric modes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("custom_cleaned_kannada", ROOT / "public/data/metrics.json", "dual"),
    ("faithful_markdown_kannada", ROOT / "experiments/faithful_markdown_kannada/metrics.json", "faithful"),
    ("faithful_markdown_maithili", ROOT / "experiments/faithful_markdown_maithili/metrics.json", "faithful"),
]


def main() -> int:
    rows = []
    missing = []
    for name, path, kind in SOURCES:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if kind == "dual":
            rows.extend([
                {"approach": name, "corpus_mode": "cleaned_plaintext", "metric": "whitespace_word", "languages": "en,hi,te,kn", "vocab_size": data["summary"]["actual_vocab_size"], "spread": data["summary"]["delta_x"], "raw_score": data["summary"]["score"], "round_trip": data["validation"]["visible_non_whitespace_round_trip_valid"]},
                {"approach": name, "corpus_mode": "cleaned_plaintext", "metric": "faithful_unit", "languages": "en,hi,te,kn", "vocab_size": data["summary"]["actual_vocab_size"], "spread": data["faithful_unit_summary"]["delta_x"], "raw_score": data["faithful_unit_summary"]["score"], "round_trip": data["validation"]["visible_non_whitespace_round_trip_valid"]},
            ])
        else:
            codes = ",".join(row["code"] for row in data["per_language"])
            rows.append({"approach": name, "corpus_mode": "faithful_markdown", "metric": "faithful_unit", "languages": codes, "vocab_size": data["vocab_size_actual"], "spread": data["summary"]["spread"], "raw_score": data["summary"]["raw_score"], "round_trip": data["round_trip_exact_valid"]})
    output = ROOT / "experiments/comparisons"
    output.mkdir(parents=True, exist_ok=True)
    report = {"warning": "Scores are comparable only when corpus_mode, metric, and language set match.", "missing_sources": missing, "rows": rows}
    (output / "approach_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "approach_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["approach", "corpus_mode", "metric", "languages", "vocab_size", "spread", "raw_score", "round_trip"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

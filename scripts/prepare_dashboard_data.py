#!/usr/bin/env python3
"""Publish selected experiment artifacts into the static Netlify data bundle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public" / "data"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_experiment(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "metrics.json"):
        shutil.copyfile(source / name, destination / name)
    manifest = source / "corpus" / "manifest.json"
    if manifest.exists():
        shutil.copyfile(manifest, destination / "corpus-manifest.json")


def main() -> int:
    baseline_path = PUBLIC_DATA / "metrics.json"
    kannada_path = ROOT / "experiments" / "faithful_markdown_kannada"
    maithili_path = ROOT / "experiments" / "faithful_markdown_maithili"
    comparison_path = ROOT / "experiments" / "comparisons" / "approach_comparison.json"
    required = [baseline_path, kannada_path / "metrics.json", kannada_path / "tokenizer.json", maithili_path / "metrics.json", maithili_path / "tokenizer.json", comparison_path]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dashboard sources: " + ", ".join(missing))

    copy_experiment(kannada_path, PUBLIC_DATA / "final" / "kannada")
    copy_experiment(maithili_path, PUBLIC_DATA / "experiments" / "maithili")
    shutil.copyfile(comparison_path, PUBLIC_DATA / "approach-comparison.json")

    baseline = read_json(baseline_path)
    kannada = read_json(kannada_path / "metrics.json")
    maithili = read_json(maithili_path / "metrics.json")
    dashboard = {
        "featured": kannada,
        "maithili": maithili,
        "baseline": {
            "built_at_utc": baseline["built_at_utc"],
            "validation": baseline["validation"],
            "whitespace_word_summary": baseline["summary"],
            "faithful_unit_summary": baseline["faithful_unit_summary"],
            "per_language": baseline["per_language"],
        },
        "artifacts": {
            "featured_tokenizer": "data/final/kannada/tokenizer.json",
            "featured_metrics": "data/final/kannada/metrics.json",
            "featured_corpus_manifest": "data/final/kannada/corpus-manifest.json",
            "maithili_tokenizer": "data/experiments/maithili/tokenizer.json",
            "maithili_metrics": "data/experiments/maithili/metrics.json",
            "comparison": "data/approach-comparison.json",
            "custom_tokenizer": "data/tokenizer.json",
            "custom_metrics": "data/metrics.json",
        },
    }
    (PUBLIC_DATA / "dashboard.json").write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Dashboard data prepared at", PUBLIC_DATA / "dashboard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

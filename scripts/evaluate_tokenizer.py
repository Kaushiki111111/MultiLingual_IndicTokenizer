#!/usr/bin/env python3
"""Re-evaluate an exported custom BPE tokenizer on a saved four-language corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_data import BPEModel, LANGUAGES, compute_score_for_field, faithful_unit_count, visible_non_whitespace, word_units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=Path("public/data/tokenizer.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("public/data/corpus"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    model = BPEModel(
        specials=list(payload["special_tokens"]),
        vocab=[str(record["token"]) for record in payload["vocab"]],
        merges=[(str(row["left"]), str(row["right"]), str(row["token"])) for row in payload["merges"]],
        initial_units=str(payload["initial_unit_mode"]),
    )
    rows = []
    for language in LANGUAGES:
        code = str(language["code"])
        text = (args.corpus_dir / f"{code}.txt").read_text(encoding="utf-8")
        encoded = model.encode_text(text)
        decoded = model.decode_tokens(encoded)
        words = len(word_units(text))
        faithful = faithful_unit_count(text)
        rows.append({
            "code": code,
            "language": str(language["name"]),
            "bpe_tokens": len(encoded),
            "word_units": words,
            "fertility_x": len(encoded) / words,
            "faithful_units": faithful,
            "faithful_fertility_x": len(encoded) / faithful,
            "round_trip_visible_valid": visible_non_whitespace(decoded) == visible_non_whitespace(text),
            "round_trip_exact_valid": decoded == text,
        })
    result = {
        "vocab_size": len(model.vocab),
        "visible_non_whitespace_round_trip_valid": all(row["round_trip_visible_valid"] for row in rows),
        "exact_whitespace_round_trip_valid": all(row["round_trip_exact_valid"] for row in rows),
        "whitespace_word_summary": compute_score_for_field(rows, "fertility_x"),
        "faithful_unit_summary": compute_score_for_field(rows, "faithful_fertility_x"),
        "per_language": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

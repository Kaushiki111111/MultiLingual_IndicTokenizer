#!/usr/bin/env python3
"""Train and evaluate one strict-10k faithful-Markdown Hugging Face BPE."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

from common import LANGUAGES, SPECIAL_TOKENS, faithful_unit_count, score_summary, sha256_text, visible_non_whitespace


def parse_weights(value: str) -> dict[str, int]:
    result = {code: int(weight) for code, weight in (item.split("=", 1) for item in value.split(","))}
    if any(weight < 1 for weight in result.values()):
        raise ValueError("all training weights must be positive integers")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", required=True, help="en=1,hi=1,te=2,kn=3")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    weights = parse_weights(args.weights)
    texts = {code: (args.corpus_dir / f"{code}.faithful.txt").read_text(encoding="utf-8") for code in weights}
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    trainer = BpeTrainer(vocab_size=10_000, min_frequency=1, special_tokens=SPECIAL_TOKENS)
    iterator = (texts[code] for code in weights for _ in range(weights[code]))
    tokenizer.train_from_iterator(iterator, length=sum(weights.values()), trainer=trainer)
    tokenizer.save(str(args.out / "tokenizer.json"))

    rows = []
    failures = []
    for code, text in texts.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        exact = decoded == text
        visible = visible_non_whitespace(decoded) == visible_non_whitespace(text)
        if not exact:
            failures.append(code)
        units = faithful_unit_count(text)
        rows.append({
            "code": code,
            "language": LANGUAGES[code]["name"],
            "wiki_title": LANGUAGES[code]["title"],
            "bpe_tokens": len(encoded.ids),
            "unique_vocab_tokens_used": len(set(encoded.ids)),
            "faithful_units": units,
            "fertility_x": len(encoded.ids) / units,
            "round_trip_exact_valid": exact,
            "round_trip_visible_valid": visible,
            "corpus_sha256": sha256_text(text),
        })
    summary = score_summary(rows)
    result = {
        "approach": "faithful_markdown_huggingface_metaspace_bpe",
        "experiment": args.name,
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "vocab_size_requested": 10_000,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "normalizer": None,
        "pretokenizer": "Metaspace(replacement=▁, prepend_scheme=never)",
        "training_weights": weights,
        "round_trip_exact_valid": not failures,
        "round_trip_visible_valid": all(row["round_trip_visible_valid"] for row in rows),
        "round_trip_failures": failures,
        "score_metric": "faithful_unit",
        "summary": summary,
        "per_language": rows,
    }
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if not failures and tokenizer.get_vocab_size() == 10_000 else 2


if __name__ == "__main__":
    raise SystemExit(main())

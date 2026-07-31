# IndicToken Lab — Multilingual India Wikipedia BPE

This repository contains a polished Netlify dashboard and reproducible tokenizer
experiments for the India Wikipedia multilingual BPE assignment. The featured
result is the faithful-Markdown English–Hindi–Telugu–Kannada tokenizer with a
10,000-entry vocabulary, exact round-trip validation, and score **26,911.69**.

The original custom approach remains published in `public/data`. Separate,
non-overwriting faithful-Markdown Hugging Face BPE experiments for Kannada and
Maithili are maintained under `approaches/` and `experiments/`. See
`approaches/README.md` for the codebase layout and `experiments/RESULTS.md` for
the current cross-approach results.

## What this version adds

This version keeps the same external tokenizer format and scoring logic, but the BPE training backend is faster and more memory-efficient:

- Each language corpus is deduplicated into a word-frequency dictionary: `unique whitespace word -> count`.
- Each word is represented internally as a tuple of integer symbol IDs: `tuple[int, ...]`, not `tuple[str, ...]`.
- BPE pairs are represented as integer pairs: `tuple[int, int]`.
- Merge rules and vocabulary are still exported as readable string tokens in `public/data/tokenizer.json`.

The tokenizer design remains:

- English, Hindi, Telugu, Kannada Wikipedia India pages.
- One shared 10,000-token BPE vocabulary including 4 special tokens.
- Grapheme / akshara-like initial units.
- Objective-guided constrained greedy merge selection.
- English constraint: `English X <= 1.2`.
- Score: `1000 / (max(X1..X4) - min(X1..X4))`.

## Dual evaluation

The original assignment score remains based on whitespace word units:

```text
word X = BPE tokens / whitespace word units
```

The build now also reports a faithful-unit comparison using the same tokenizer
and the same English, Hindi, Telugu, and Kannada corpus:

```text
faithful unit = one contiguous Unicode letter/mark/number run
                OR one visible non-whitespace punctuation/symbol character
faithful X = BPE tokens / faithful units
faithful score = 1000 / (max faithful X - min faithful X)
```

The custom word-boundary tokenizer normalizes whitespace, so validation checks
the assignment's visible-text contract: decoding must preserve exactly the same
non-whitespace characters in the same order. Exact whitespace equality is also
reported separately. Generated `evaluation.json` contains both score summaries
and the per-language round-trip results.

Re-evaluate the exported tokenizer against its saved corpus independently:

```bash
python scripts/evaluate_tokenizer.py
```

## Run locally

From this folder:

```bash
python scripts/prepare_dashboard_data.py
python -m http.server 5173 --directory public
```

Open:

```text
http://localhost:5173/
```

Tokenizer JSON download URL locally:

```text
http://localhost:5173/data/final/kannada/tokenizer.json
```

## Manual Netlify deploy

1. Prepare the static dashboard bundle:

```bash
python scripts/prepare_dashboard_data.py
```

2. Upload the entire `public` folder to Netlify.

Your final submission links will be:

```text
Widget URL:
https://YOUR-SITE.netlify.app/

Tokenizer.json download URL:
https://YOUR-SITE.netlify.app/data/final/kannada/tokenizer.json
```

## GitHub + Netlify deploy

Build command:

```text
python3 scripts/prepare_dashboard_data.py
```

Publish directory:

```text
public
```

# Tokenizer approaches and experiments

The repository keeps each corpus/tokenizer methodology separate so results remain
auditable and comparable.

## Approaches

1. `custom_cleaned_grapheme_bpe`
   - Existing implementation in `scripts/build_data.py`.
   - Wikipedia TextExtracts plain text, cleaned article body.
   - Custom grapheme/akshara BPE and constrained-greedy optimization.
   - Primary denominator: whitespace words.
   - Published artifacts remain in `public/data/`.

2. `faithful_markdown_hf_bpe`
   - Implementation in `approaches/faithful_markdown_hf_bpe/`.
   - Wikipedia REST HTML converted to faithful Markdown.
   - Standard Hugging Face BPE with reversible Metaspace encoding.
   - Primary denominator: faithful units.
   - Kannada and Maithili runs are written to different directories under
     `experiments/`; they never overwrite `public/data/`.

## Experiment layout

```text
experiments/
  faithful_markdown_kannada/
    corpus/
    tokenizer.json
    metrics.json
  faithful_markdown_maithili/
    corpus/
    tokenizer.json
    metrics.json
  comparisons/
    approach_comparison.json
    approach_comparison.csv
```

Corpus snapshots, metadata, tokenizer files, metrics, and configurations should
be committed when preparing the future GitHub repository. Avoid replacing one
approach with another; add a new named experiment instead.

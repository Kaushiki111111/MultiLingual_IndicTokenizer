# Faithful-Markdown Hugging Face BPE

This approach is intentionally separate from the existing custom cleaned-text
trainer. It supports two reproducible language profiles without overwriting the
published baseline.

## Install

```bash
pip install -r approaches/faithful_markdown_hf_bpe/requirements.txt
```

## Kannada profile

```bash
python approaches/faithful_markdown_hf_bpe/build_corpus.py --languages en,hi,te,kn --out experiments/faithful_markdown_kannada/corpus
python approaches/faithful_markdown_hf_bpe/train_evaluate.py --corpus-dir experiments/faithful_markdown_kannada/corpus --out experiments/faithful_markdown_kannada --name faithful_markdown_kannada --weights en=1,hi=1,te=2,kn=3
```

## Maithili profile

```bash
python approaches/faithful_markdown_hf_bpe/build_corpus.py --languages en,hi,te,mai --out experiments/faithful_markdown_maithili/corpus
python approaches/faithful_markdown_hf_bpe/train_evaluate.py --corpus-dir experiments/faithful_markdown_maithili/corpus --out experiments/faithful_markdown_maithili --name faithful_markdown_maithili --weights en=3,hi=4,te=4,mai=2
```

## Comparison report

```bash
python scripts/compare_approaches.py
```

The Markdown corpus snapshots are used unchanged for training and evaluation.
The tokenizer has no normalizer and must pass exact `decode(encode(text))`
validation on every full corpus file.

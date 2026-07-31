# Experiment results

Generated from frozen corpus snapshots on 2026-08-01 (Asia/Calcutta). Do not
compare scores across different corpus modes, metrics, or language sets as if
they were the same experiment.

| Approach | Corpus | Languages | Metric | Spread | Raw score | Round trip |
| --- | --- | --- | --- | ---: | ---: | --- |
| Custom grapheme constrained BPE | cleaned plaintext | en, hi, te, kn | whitespace words | 0.752728 | 1,328.50 | visible characters pass |
| Custom grapheme constrained BPE | cleaned plaintext | en, hi, te, kn | faithful units | 0.693457 | 1,442.05 | visible characters pass |
| Hugging Face Metaspace BPE | faithful Markdown | en, hi, te, kn | faithful units | 0.037159 | 26,911.69 | exact pass |
| Hugging Face Metaspace BPE | faithful Markdown | en, hi, te, mai | faithful units | 0.142680 | 7,008.69 | exact pass |

## Faithful-Markdown Kannada

Weights: `en=1, hi=1, te=2, kn=3`

| Language | Tokens | Faithful units | Fertility |
| --- | ---: | ---: | ---: |
| English | 114,464 | 180,338 | 0.634719 |
| Hindi | 55,784 | 83,027 | 0.671878 |
| Telugu | 21,846 | 33,730 | 0.647673 |
| Kannada | 7,458 | 11,409 | 0.653694 |

## Faithful-Markdown Maithili

Weights: `en=3, hi=4, te=4, mai=2`

| Language | Tokens | Faithful units | Fertility |
| --- | ---: | ---: | ---: |
| English | 110,437 | 180,338 | 0.612389 |
| Hindi | 50,160 | 83,027 | 0.604141 |
| Telugu | 23,524 | 33,730 | 0.697421 |
| Maithili | 3,876 | 5,190 | 0.746821 |

Machine-readable comparisons are in `comparisons/approach_comparison.json` and
`comparisons/approach_comparison.csv`.

#!/usr/bin/env python3
"""
Build India Wikipedia multilingual BPE tokenizer data for the widget.

Design choices:
- Languages: English, Hindi, Telugu, Kannada.
- Corpus: Wikipedia plain-text extracts for each article, cleaned to article body + headings.
- Tokenizer: shared word-boundary BPE tokenizer, unweighted multilingual corpus.
- Initial units: Unicode grapheme/akshara-like clusters rather than raw Unicode codepoints.
- Vocab budget: 10,000 total tokens including 4 special tokens.
- Raw diagnostic mode: train on the first 80% of each article and evaluate on the held-out last 20%.
  This prevents a tiny/custom corpus from being memorised as whole-word tokens, which is what causes
  the misleading all-languages fertility = 1.0 result.
- Fertility ratio X: BPE token count / whitespace word-unit count. Lower is better.
- Assignment score: 1000 / (max(X) - min(X)).

No third-party packages are required.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import heapq
import html
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Sequence, Tuple

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
WORD_BOUNDARY = "▁"
DEFAULT_VOCAB_SIZE = 10_000
INITIAL_UNITS = "grapheme"

LANGUAGES = [
    {
        "code": "en",
        "name": "English",
        "title": "India",
        "url": "https://en.wikipedia.org/wiki/India",
        "api": "https://en.wikipedia.org/w/api.php",
        "stop_headings": ["See also", "Notes", "References", "Further reading", "External links"],
    },
    {
        "code": "hi",
        "name": "Hindi",
        "title": "भारत",
        "url": "https://hi.wikipedia.org/wiki/भारत",
        "api": "https://hi.wikipedia.org/w/api.php",
        "stop_headings": ["इन्हें भी देखें", "सन्दर्भ", "संदर्भ", "टिप्पणी सूची", "बाहरी कड़ियाँ", "बाहरी कड़ियां"],
    },
    {
        "code": "te",
        "name": "Telugu",
        "title": "భారతదేశం",
        "url": "https://te.wikipedia.org/wiki/భారతదేశం",
        "api": "https://te.wikipedia.org/w/api.php",
        "stop_headings": ["ఇవి కూడా చూడండి", "చూడండి", "మూలాలు", "బయటి లింకులు", "బయటి లంకెలు", "ఆధారాలు"],
    },
    {
        "code": "kn",
        "name": "Kannada",
        "title": "ಭಾರತ",
        "url": "https://kn.wikipedia.org/wiki/ಭಾರತ",
        "api": "https://kn.wikipedia.org/w/api.php",
        "stop_headings": ["ಇವನ್ನೂ ನೋಡಿ", "ಉಲ್ಲೇಖಗಳು", "ಹೊರಗಿನ ಸಂಪರ್ಕಗಳು", "ಬಾಹ್ಯ ಕೊಂಡಿಗಳು", "ಆಧಾರಗಳು"],
    },
]

# Small offline samples used only when the build machine has no network.
# Netlify or any normal internet-connected run will fetch the live Wikipedia articles.
FALLBACK_TEXT = {
    "en": """
India
India, officially the Republic of India, is a country in South Asia. It is the seventh-largest country by area and the most populous country in the world. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan, China, Nepal, Bhutan, Bangladesh and Myanmar. Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago. India has been a federal republic since 1950, governed through a democratic parliamentary system. It is a pluralistic, multilingual and multi-ethnic society. The Indian economy has become a fast-growing major economy and a hub for information technology, with an expanding middle class.
Etymology
The English proper noun India derives from Classical Latin India. The term Bharat is used in many Indian languages. Hindustan is a Middle Persian name for India that became popular by the thirteenth century.
History
The Indus Valley Civilisation flourished during the third millennium BCE. Early political consolidations gave rise to the Maurya and Gupta empires. The Mughal Empire ushered in economic expansion and left a rich architectural legacy. British Crown rule began in 1858, and a nationalist movement ended British rule in 1947.
Geography
India accounts for the bulk of the Indian subcontinent, lying atop the Indian tectonic plate. The country has varied landscapes, from the Himalayas to coastal plains, deserts, forests and river systems.
""",
    "hi": """
भारत
भारत आधिकारिक नाम भारत गणराज्य दक्षिण एशिया में स्थित भारतीय उपमहाद्वीप का सबसे बड़ा देश है। भारत भौगोलिक दृष्टि से विश्व का सातवाँ सबसे बड़ा देश है, जबकि जनसंख्या के दृष्टिकोण से विश्व का सबसे बड़ा देश है। भारत के पश्चिम में पाकिस्तान, उत्तर में चीन, नेपाल और भूटान तथा पूर्व में बांग्लादेश और म्यान्मार स्थित हैं। हिन्द महासागर में इसके दक्षिण में श्रीलंका और मालदीव हैं। भारत 1950 से एक संघीय गणराज्य है और संसदीय लोकतांत्रिक प्रणाली से शासित है। यह बहुभाषी, बहुधार्मिक और विविध समाज वाला देश है। भारत की अर्थव्यवस्था सूचना प्रौद्योगिकी, सेवा क्षेत्र, कृषि और उद्योग के कारण तेज़ी से बढ़ती अर्थव्यवस्था मानी जाती है।
नामोत्पत्ति
भारत के दो आधिकारिक नाम हैं भारत और इंडिया। इंडिया नाम की उत्पत्ति सिंधु नदी के अंग्रेज़ी नाम इंडस से हुई है। भारत नाम प्राचीन भारतीय परंपराओं और ग्रंथों में मिलता है।
इतिहास
सिंधु घाटी सभ्यता भारतीय उपमहाद्वीप की प्राचीन सभ्यताओं में से एक थी। मौर्य और गुप्त साम्राज्यों ने प्राचीन भारत में राजनीतिक एकता और सांस्कृतिक विकास को बढ़ावा दिया। मध्यकालीन भारत में अनेक राज्य, सल्तनत और मुगल साम्राज्य उभरे। ब्रिटिश शासन के विरुद्ध स्वतंत्रता आन्दोलन चला और 1947 में भारत स्वतंत्र हुआ।
भूगोल
भारत में हिमालय, गंगा का मैदान, थार मरुस्थल, दक्कन का पठार, तटीय मैदान और अनेक नदी प्रणालियाँ हैं।
""",
    "te": """
భారతదేశం
భారతదేశం దక్షిణ ఆసియాలో ఉన్న పెద్ద దేశం. ఇది విస్తీర్ణం పరంగా ప్రపంచంలో ఏడవ పెద్ద దేశం మరియు జనాభా పరంగా ప్రపంచంలో అతి పెద్ద దేశాలలో ఒకటి. భారతదేశం దక్షిణాన హిందూ మహాసముద్రం, పడమరన అరేబియా సముద్రం, తూర్పున బంగాళాఖాతంతో చుట్టుముట్టబడి ఉంది. దీనికి పాకిస్తాన్, చైనా, నేపాల్, భూటాన్, బంగ్లాదేశ్ మరియు మయన్మార్ దేశాలతో భూసరిహద్దులు ఉన్నాయి. భారతదేశం 1950 నుండి సమాఖ్య గణరాజ్యంగా ఉంది. ఇక్కడ పార్లమెంటరీ ప్రజాస్వామ్య విధానం ఉంది. భారతదేశం అనేక భాషలు, మతాలు, సంస్కృతులు కలిగిన విభిన్న సమాజం. ఆర్థిక వ్యవస్థలో వ్యవసాయం, పరిశ్రమలు, సేవలు మరియు సమాచార సాంకేతిక రంగం ముఖ్యమైనవి.
పేరుపుట్టుక
భారతదేశం అనే పేరు ప్రాచీన భారతీయ సంప్రదాయాలలో కనిపిస్తుంది. ఇండియా అనే పేరు సింధు నది పేరుతో సంబంధం కలిగి ఉంది. భారత్ అనే పేరు అనేక భారతీయ భాషలలో వాడబడుతుంది.
చరిత్ర
సింధు లోయ నాగరికత భారత ఉపఖండంలోని ప్రాచీన నాగరికతలలో ఒకటి. మౌర్య మరియు గుప్త సామ్రాజ్యాలు ప్రాచీన భారతదేశంలో రాజకీయ మరియు సాంస్కృతిక వికాసానికి తోడ్పడ్డాయి. మధ్యయుగ కాలంలో అనేక రాజ్యాలు మరియు సామ్రాజ్యాలు ఏర్పడ్డాయి. బ్రిటిష్ పాలనకు వ్యతిరేకంగా స్వాతంత్ర్య ఉద్యమం జరిగి 1947లో భారతదేశం స్వతంత్రం అయింది.
భూగోళం
భారతదేశంలో హిమాలయ పర్వతాలు, గంగా మైదానం, థార్ ఎడారి, దక్కను పీఠభూమి, తీరప్రాంతాలు మరియు అనేక నదులు ఉన్నాయి.
""",
    "kn": """
ಭಾರತ
ಭಾರತ ಅಧಿಕೃತವಾಗಿ ಭಾರತ ಗಣರಾಜ್ಯ ದಕ್ಷಿಣ ಏಷ್ಯಾದ ಪ್ರಮುಖ ದೇಶವಾಗಿದೆ. ಇದು ವಿಸ್ತೀರ್ಣದ ಆಧಾರದ ಮೇಲೆ ವಿಶ್ವದ ಏಳನೇ ದೊಡ್ಡ ದೇಶವಾಗಿದ್ದು, ಜನಸಂಖ್ಯೆಯ ಆಧಾರದ ಮೇಲೆ ವಿಶ್ವದ ಅತ್ಯಂತ ದೊಡ್ಡ ದೇಶಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ. ಭಾರತ ದಕ್ಷಿಣದಲ್ಲಿ ಹಿಂದೂ ಮಹಾಸಾಗರ, ಪಶ್ಚಿಮದಲ್ಲಿ ಅರಬ್ಬಿ ಸಮುದ್ರ ಮತ್ತು ಪೂರ್ವದಲ್ಲಿ ಬೆಂಗಾಲ್ ಕೊಲ್ಲಿಯಿಂದ ಸುತ್ತುವರಿದಿದೆ. ಪಾಕಿಸ್ತಾನ, ಚೀನಾ, ನೇಪಾಳ, ಭೂತಾನ್, ಬಾಂಗ್ಲಾದೇಶ ಮತ್ತು ಮ್ಯಾನ್ಮಾರ್ ದೇಶಗಳೊಂದಿಗೆ ಭಾರತ ಗಡಿಯನ್ನು ಹಂಚಿಕೊಂಡಿದೆ. ಭಾರತ 1950ರಿಂದ ಸಂಘೀಯ ಗಣರಾಜ್ಯವಾಗಿದ್ದು, ಸಂಸತ್ತಿನ ಪ್ರಜಾಪ್ರಭುತ್ವ ವ್ಯವಸ್ಥೆಯಿಂದ ಆಡಳಿತ ನಡೆಸಲ್ಪಡುತ್ತದೆ. ಭಾರತವು ಬಹುಭಾಷಾ, ಬಹುಧಾರ್ಮಿಕ ಮತ್ತು ವೈವಿಧ್ಯಮಯ ಸಮಾಜವಾಗಿದೆ. ಭಾರತದ ಆರ್ಥಿಕತೆಯಲ್ಲಿ ಕೃಷಿ, ಕೈಗಾರಿಕೆ, ಸೇವಾ ವಲಯ ಮತ್ತು ಮಾಹಿತಿ ತಂತ್ರಜ್ಞಾನ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತವೆ.
ಹೆಸರಿನ ಉಗಮ
ಭಾರತ ಎಂಬ ಹೆಸರು ಪ್ರಾಚೀನ ಭಾರತೀಯ ಪರಂಪರೆಗಳಲ್ಲಿ ಕಂಡುಬರುತ್ತದೆ. ಇಂಡಿಯಾ ಎಂಬ ಹೆಸರು ಸಿಂಧು ನದಿಯ ಹೆಸರಿನಿಂದ ಬಂದಿದೆ. ಭಾರತ ಎಂಬ ಪದವನ್ನು ಅನೇಕ ಭಾರತೀಯ ಭಾಷೆಗಳಲ್ಲಿ ಬಳಸಲಾಗುತ್ತದೆ.
ಇತಿಹಾಸ
ಸಿಂಧು ಕಣಿವೆ ನಾಗರಿಕತೆ ಭಾರತೀಯ ಉಪಖಂಡದ ಪ್ರಾಚೀನ ನಾಗರಿಕತೆಯಾಗಿದೆ. ಮೌರ್ಯ ಮತ್ತು ಗುಪ್ತ ಸಾಮ್ರಾಜ್ಯಗಳು ಪ್ರಾಚೀನ ಭಾರತದಲ್ಲಿ ರಾಜಕೀಯ ಮತ್ತು ಸಾಂಸ್ಕೃತಿಕ ಬೆಳವಣಿಗೆಗೆ ಕಾರಣವಾದವು. ಮಧ್ಯಯುಗದಲ್ಲಿ ಅನೇಕ ರಾಜ್ಯಗಳು ಮತ್ತು ಸಾಮ್ರಾಜ್ಯಗಳು ಹುಟ್ಟಿಕೊಂಡವು. ಬ್ರಿಟಿಷ್ ಆಡಳಿತದ ವಿರುದ್ಧ ಸ್ವಾತಂತ್ರ್ಯ ಚಳವಳಿ ನಡೆದಿದ್ದು, 1947ರಲ್ಲಿ ಭಾರತ ಸ್ವತಂತ್ರವಾಯಿತು.
ಭೂಗೋಳ
ಭಾರತದಲ್ಲಿ ಹಿಮಾಲಯ ಪರ್ವತಗಳು, ಗಂಗಾ ಸಮತಟ, ಥಾರ್ ಮರುಭೂಮಿ, ದಕ್ಕನ್ ಪೀಠಭೂಮಿ, ಕರಾವಳಿ ಪ್ರದೇಶಗಳು ಮತ್ತು ಅನೇಕ ನದಿಗಳು ಇವೆ.
""",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    # Remove common citation artifacts while preserving useful punctuation and native scripts.
    text = re.sub(r"\[[0-9a-zA-Z]+\]", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_extract(text: str, stop_headings: Sequence[str]) -> str:
    text = normalize_text(text)
    out_lines: List[str] = []
    stop_set = {s.strip().casefold() for s in stop_headings}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out_lines.append("")
            continue
        simplified = re.sub(r"^=+|=+$", "", line).strip().casefold()
        if simplified in stop_set or any(simplified.startswith(x + " ") for x in stop_set):
            break
        # Remove edit/nav remnants if the extract endpoint includes them.
        if simplified in {"edit", "source edit", "संपादित करें", "ಮೂಲ ಸಂಪಾದಿಸಿ", "మార్చు"}:
            continue
        out_lines.append(line)
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def fetch_wikipedia_extract(lang: Dict[str, object], timeout: int = 25) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": str(lang["title"]),
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    }
    url = str(lang["api"]) + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "IndiaBPEAssignmentWidget/1.0 (reproducible student project)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", [])
    if not pages or "extract" not in pages[0]:
        raise RuntimeError(f"No extract returned for {lang['name']} / {lang['title']}")
    return clean_extract(pages[0]["extract"], lang["stop_headings"])


def word_units(text: str) -> List[str]:
    # Whitespace word units are used for the denominator because Indian-language words
    # do not use the same punctuation/segmentation patterns across scripts.
    return [w for w in re.split(r"\s+", text.strip()) if w]


def faithful_unit_count(text: str) -> int:
    """Count letter/mark/number runs and individual visible symbols."""
    count = 0
    in_alnum_run = False
    for ch in text:
        if unicodedata.category(ch)[0] in {"L", "M", "N"}:
            if not in_alnum_run:
                count += 1
                in_alnum_run = True
        else:
            in_alnum_run = False
            if not ch.isspace():
                count += 1
    return count


def visible_non_whitespace(text: str) -> str:
    """Return the character sequence covered by the visible-text fidelity rule."""
    return "".join(ch for ch in text if not ch.isspace())


def split_for_raw_diagnostic(text: str, train_fraction: float = 0.80) -> Tuple[str, str]:
    """Chronological paragraph/line split: train on earlier text, evaluate on unseen later text.

    If we train and evaluate BPE on the exact same short article with a 10k vocabulary,
    BPE can learn many entire whitespace words. That produces fertility=1.0, which is
    a memorisation artefact rather than a useful tokenizer-fertility measurement.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n", text) if b.strip()]
    if len(blocks) < 2:
        words = word_units(text)
        cut = max(1, min(len(words) - 1, int(len(words) * train_fraction))) if len(words) > 1 else len(words)
        return " ".join(words[:cut]), " ".join(words[cut:]) or " ".join(words[-1:])
    cut = int(len(blocks) * train_fraction)
    cut = max(1, min(len(blocks) - 1, cut))
    return "\n".join(blocks[:cut]), "\n".join(blocks[cut:])


def is_unicode_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def is_joiner(ch: str) -> bool:
    return ch in {"\u200c", "\u200d"}


def is_virama(ch: str) -> bool:
    # Covers Devanagari halant, Telugu virama, Kannada virama, and other Indic viramas.
    return "VIRAMA" in unicodedata.name(ch, "")


def grapheme_clusters(text: str) -> List[str]:
    """Return a small, dependency-free approximation of Unicode grapheme clusters.

    For Latin text, this is effectively character + combining marks.
    For Indic scripts, it keeps base letters with dependent vowel signs/diacritics and
    attaches virama/ZWJ conjunct sequences to the same cluster. This gives us
    akshara-like initial BPE units and avoids unfairly splitting Telugu/Kannada/Hindi
    words into many raw Unicode codepoints before BPE even starts.
    """
    text = unicodedata.normalize("NFC", text)
    clusters: List[str] = []
    current = ""
    force_attach_next_base = False

    for ch in text:
        cat = unicodedata.category(ch)
        attach = False
        if not current:
            attach = True
        elif is_unicode_mark(ch) or is_joiner(ch):
            attach = True
        elif force_attach_next_base:
            attach = True
        elif cat in {"Cf"}:
            attach = True

        if attach:
            current += ch
        else:
            clusters.append(current)
            current = ch

        # If a cluster ends with virama or joiner, the next base consonant belongs to the same cluster.
        force_attach_next_base = is_virama(ch) or is_joiner(ch)

    if current:
        clusters.append(current)
    return clusters


def initial_units_for_word(word: str, mode: str = INITIAL_UNITS) -> List[str]:
    if mode == "grapheme":
        return grapheme_clusters(word)
    if mode == "codepoint":
        return list(word)
    raise ValueError(f"Unsupported initial unit mode: {mode}")


def initial_baseline_tokens(text: str, mode: str = INITIAL_UNITS) -> int:
    """Number of initial symbols before BPE merges, with one ▁ boundary per word."""
    return sum(1 + len(initial_units_for_word(w, mode=mode)) for w in word_units(text))


def initial_symbols_for_word(word: str, mode: str = INITIAL_UNITS) -> Tuple[str, ...]:
    # SentencePiece-like boundary marker lets BPE learn beginning-of-word pieces.
    return tuple([WORD_BOUNDARY] + initial_units_for_word(word, mode=mode))


def merge_symbols(symbols: Sequence[str], pair: Tuple[str, str], merged: str) -> Tuple[str, ...]:
    out: List[str] = []
    i = 0
    n = len(symbols)
    a, b = pair
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def pair_counter(symbols: Sequence[str]) -> Counter[Tuple[str, str]]:
    c: Counter[Tuple[str, str]] = collections.Counter()
    for i in range(len(symbols) - 1):
        c[(symbols[i], symbols[i + 1])] += 1
    return c


@dataclass
class BPEModel:
    specials: List[str]
    vocab: List[str]
    merges: List[Tuple[str, str, str]]
    initial_units: str = INITIAL_UNITS

    def merge_ranks(self) -> Dict[Tuple[str, str], Tuple[int, str]]:
        return {(a, b): (i, merged) for i, (a, b, merged) in enumerate(self.merges)}

    def encode_word_with_ranks(self, word: str, ranks: Dict[Tuple[str, str], Tuple[int, str]]) -> List[str]:
        symbols = list(initial_symbols_for_word(word, mode=self.initial_units))
        if len(symbols) < 2:
            return symbols
        while True:
            best_idx = -1
            best_rank = math.inf
            best_merged = None
            for i in range(len(symbols) - 1):
                key = (symbols[i], symbols[i + 1])
                found = ranks.get(key)
                if found is not None and found[0] < best_rank:
                    best_rank = found[0]
                    best_merged = found[1]
                    best_idx = i
            if best_idx < 0 or best_merged is None:
                break
            symbols = symbols[:best_idx] + [best_merged] + symbols[best_idx + 2 :]
        return symbols

    def encode_word(self, word: str) -> List[str]:
        return self.encode_word_with_ranks(word, self.merge_ranks())

    def encode_text(self, text: str) -> List[str]:
        encoded: List[str] = []
        ranks = self.merge_ranks()
        cache: Dict[str, List[str]] = {}
        for word in word_units(text):
            pieces = cache.get(word)
            if pieces is None:
                pieces = self.encode_word_with_ranks(word, ranks)
                cache[word] = pieces
            encoded.extend(pieces)
        return encoded

    def decode_tokens(self, tokens: Sequence[str]) -> str:
        """Decode symbols correctly, including collision-renamed merge tokens."""
        expansions = {merged: (left, right) for left, right, merged in self.merges}
        cache: Dict[str, str] = {}

        def expand(token: str) -> str:
            cached = cache.get(token)
            if cached is not None:
                return cached
            pair = expansions.get(token)
            value = token if pair is None else expand(pair[0]) + expand(pair[1])
            cache[token] = value
            return value

        return "".join(expand(token) for token in tokens).replace(WORD_BOUNDARY, " ")

    def decode_text(self, text: str) -> str:
        return self.decode_tokens(self.encode_text(text))


def train_bpe(texts: Sequence[str], vocab_size: int = DEFAULT_VOCAB_SIZE, initial_units: str = INITIAL_UNITS) -> BPEModel:
    word_counts: Counter[str] = collections.Counter()
    for text in texts:
        word_counts.update(word_units(text))

    # Deduplicating words makes BPE training significantly faster while preserving frequencies.
    entries: List[Tuple[str, ...]] = [initial_symbols_for_word(w, mode=initial_units) for w in word_counts.keys()]
    freqs: List[int] = list(word_counts.values())

    vocab: List[str] = list(SPECIAL_TOKENS)
    seen_vocab = set(vocab)
    for sym in sorted({s for entry in entries for s in entry}):
        if sym not in seen_vocab and len(vocab) < vocab_size:
            vocab.append(sym)
            seen_vocab.add(sym)

    pair_counts: Counter[Tuple[str, str]] = collections.Counter()
    pair_to_words: Dict[Tuple[str, str], set[int]] = collections.defaultdict(set)
    word_pair_counts: List[Counter[Tuple[str, str]]] = []

    for idx, symbols in enumerate(entries):
        pc = pair_counter(symbols)
        word_pair_counts.append(pc)
        for pair, count in pc.items():
            pair_counts[pair] += count * freqs[idx]
            pair_to_words[pair].add(idx)

    heap: List[Tuple[int, Tuple[str, str]]] = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges: List[Tuple[str, str, str]] = []
    used_merge_names = set(seen_vocab)

    while len(vocab) < vocab_size and heap:
        neg_count, pair = heapq.heappop(heap)
        current_count = pair_counts.get(pair, 0)
        if current_count <= 0 or -neg_count != current_count:
            continue
        a, b = pair
        merged = a + b
        # Collision guard. Very rare, but possible if a merged string equals an existing char/token.
        if merged in used_merge_names:
            suffix = 2
            candidate = f"{merged}⟨{suffix}⟩"
            while candidate in used_merge_names:
                suffix += 1
                candidate = f"{merged}⟨{suffix}⟩"
            merged = candidate
        affected = list(pair_to_words.get(pair, set()))
        if not affected:
            pair_counts[pair] = 0
            continue

        vocab.append(merged)
        seen_vocab.add(merged)
        used_merge_names.add(merged)
        merges.append((a, b, merged))

        touched_pairs: set[Tuple[str, str]] = set()
        for idx in affected:
            old_symbols = entries[idx]
            if pair not in word_pair_counts[idx]:
                continue
            f = freqs[idx]
            # Remove old pair contributions for this word.
            for old_pair, c in word_pair_counts[idx].items():
                pair_counts[old_pair] -= c * f
                touched_pairs.add(old_pair)
                if idx in pair_to_words.get(old_pair, set()):
                    pair_to_words[old_pair].discard(idx)
            new_symbols = merge_symbols(old_symbols, pair, merged)
            entries[idx] = new_symbols
            new_pc = pair_counter(new_symbols)
            word_pair_counts[idx] = new_pc
            for new_pair, c in new_pc.items():
                pair_counts[new_pair] += c * f
                pair_to_words[new_pair].add(idx)
                touched_pairs.add(new_pair)

        for p in touched_pairs:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))

    # Keep the assignment contract: exactly `vocab_size` entries overall.
    # If the small corpus is exhausted before the budget, reserve unused tokens.
    unused_id = 0
    while len(vocab) < vocab_size:
        tok = f"<unused_{unused_id:05d}>"
        unused_id += 1
        if tok not in seen_vocab:
            vocab.append(tok)
            seen_vocab.add(tok)

    return BPEModel(specials=list(SPECIAL_TOKENS), vocab=vocab, merges=merges, initial_units=initial_units)



LANG_CODE_ORDER = [str(l["code"]) for l in LANGUAGES]
LANG_NAME_BY_CODE = {str(l["code"]): str(l["name"]) for l in LANGUAGES}
LANG_INDEX_BY_CODE = {code: i for i, code in enumerate(LANG_CODE_ORDER)}


def train_objective_guided_bpe(
    corpus_by_code: Dict[str, str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    initial_units: str = INITIAL_UNITS,
    english_max_x: float = 1.20,
    english_boost: float = 22.0,
    fairness_strength: float = 5.0,
    worst_boost: float = 3.0,
    min_pressure: float = 0.02,
    rebuild_every: int = 18,
    checkpoint_every: int = 500,
    constraint_start_merge: int = 0,
    config_name: str = "objective_guided",
) -> Tuple[BPEModel, Dict[str, object]]:
    """Train one shared BPE tokenizer with an objective-guided greedy merge rule.

    Standard BPE chooses the most frequent adjacent pair globally. That optimizes average
    compression, but it can under-serve small Indic articles and it does not enforce the
    assignment's English X <= 1.2 constraint. This trainer still learns legal BPE merge rules,
    but scores each possible merge by its estimated reduction in language-level fertility:

        score(pair) = sum_language pressure(language) * pair_count(language) / word_units(language)

    Pressures are recomputed during training. English receives high pressure only while its
    fertility is above 1.2. Languages far above the current minimum fertility receive fairness
    pressure. Languages already near the minimum receive little pressure so we do not push them
    even lower and enlarge max(X)-min(X).
    """
    lang_codes = list(LANG_CODE_ORDER)
    n_langs = len(lang_codes)
    word_unit_counts = {code: len(word_units(corpus_by_code[code])) for code in lang_codes}

    word_entries: List[Tuple[str, int, Tuple[str, ...]]] = []
    for code in lang_codes:
        wc = collections.Counter(word_units(corpus_by_code[code]))
        for word, freq in wc.items():
            word_entries.append((code, int(freq), initial_symbols_for_word(word, mode=initial_units)))

    entries: List[Tuple[str, ...]] = [e[2] for e in word_entries]
    freqs: List[int] = [e[1] for e in word_entries]
    entry_lang_indices: List[int] = [LANG_INDEX_BY_CODE[e[0]] for e in word_entries]

    vocab: List[str] = list(SPECIAL_TOKENS)
    seen_vocab = set(vocab)
    for sym in sorted({s for entry in entries for s in entry}):
        if sym not in seen_vocab and len(vocab) < vocab_size:
            vocab.append(sym)
            seen_vocab.add(sym)

    current_tokens = {code: 0 for code in lang_codes}
    for symbols, freq, lang_idx in zip(entries, freqs, entry_lang_indices):
        current_tokens[lang_codes[lang_idx]] += len(symbols) * freq

    def zero_counts() -> List[int]:
        return [0 for _ in range(n_langs)]

    pair_counts_by_lang: Dict[Tuple[str, str], List[int]] = collections.defaultdict(zero_counts)
    pair_to_words: Dict[Tuple[str, str], set[int]] = collections.defaultdict(set)
    word_pair_counts: List[Counter[Tuple[str, str]]] = []

    for idx_entry, symbols in enumerate(entries):
        pc = pair_counter(symbols)
        word_pair_counts.append(pc)
        f = freqs[idx_entry]
        lang_idx = entry_lang_indices[idx_entry]
        for pair, count in pc.items():
            pair_counts_by_lang[pair][lang_idx] += count * f
            pair_to_words[pair].add(idx_entry)

    def x_by_code() -> Dict[str, float]:
        return {code: (current_tokens[code] / word_unit_counts[code] if word_unit_counts[code] else 0.0) for code in lang_codes}

    def compute_pressures() -> List[float]:
        xs = x_by_code()
        max_x = max(xs.values())
        min_x = min(xs.values())
        delta = max(max_x - min_x, 1e-9)
        pressures: List[float] = []
        for code in lang_codes:
            x = xs[code]
            normalized_gap = max(0.0, (x - min_x) / delta)
            pressure = min_pressure + fairness_strength * (normalized_gap ** 1.35)
            if x >= max_x - 1e-12:
                pressure += worst_boost

            if code == "en":
                if x > english_max_x and len(merges) >= constraint_start_merge:
                    # Constraint pressure rises sharply as English remains above 1.2.
                    # constraint_start_merge lets us do fair BPE first, then rescue English late.
                    over = max(0.0, x - english_max_x)
                    pressure += english_boost * (1.0 + min(2.5, over / 0.20))
                elif x <= min_x + 0.03:
                    # Once English passes, do not keep making it the minimum.
                    pressure *= 0.12
            else:
                # Languages already at the floor should not be compressed further unless they become max again.
                if x <= min_x + 0.04:
                    pressure *= 0.18
            pressures.append(max(min_pressure, pressure))
        return pressures

    def pair_score(pair: Tuple[str, str], pressures: Sequence[float]) -> float:
        counts = pair_counts_by_lang.get(pair)
        if not counts:
            return 0.0
        score = 0.0
        total = 0
        for i, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            total += cnt
            code = lang_codes[i]
            denom = max(1, word_unit_counts[code])
            # cnt/denom is estimated direct reduction in that language's X.
            score += float(pressures[i]) * (cnt / denom)
        if total <= 0:
            return 0.0
        # Tiny tie-breaker for broad utility without letting large languages dominate.
        return score + 1e-9 * math.log1p(total)

    def rebuild_heap() -> Tuple[List[Tuple[float, int, Tuple[str, str]]], List[float]]:
        pressures = compute_pressures()
        heap: List[Tuple[float, int, Tuple[str, str]]] = []
        for pair, counts in pair_counts_by_lang.items():
            total = sum(counts)
            if total <= 0:
                continue
            sc = pair_score(pair, pressures)
            if sc > 0:
                heap.append((-sc, -total, pair))
        heapq.heapify(heap)
        return heap, pressures

    merges: List[Tuple[str, str, str]] = []
    used_merge_names = set(seen_vocab)
    progress: List[Dict[str, object]] = []
    heap, pressures = rebuild_heap()
    steps_since_rebuild = 0
    start_time = time.time()

    while len(vocab) < vocab_size:
        if not heap or steps_since_rebuild >= rebuild_every:
            heap, pressures = rebuild_heap()
            steps_since_rebuild = 0
            if not heap:
                break

        _, _, pair = heapq.heappop(heap)
        counts_now = pair_counts_by_lang.get(pair)
        if not counts_now or sum(counts_now) <= 0:
            continue
        affected = list(pair_to_words.get(pair, set()))
        if not affected:
            # Stale pair-to-words entry.
            pair_counts_by_lang[pair] = zero_counts()
            continue

        a, b = pair
        merged = a + b
        if merged in used_merge_names:
            suffix = 2
            candidate = f"{merged}⟨{suffix}⟩"
            while candidate in used_merge_names:
                suffix += 1
                candidate = f"{merged}⟨{suffix}⟩"
            merged = candidate

        vocab.append(merged)
        seen_vocab.add(merged)
        used_merge_names.add(merged)
        merges.append((a, b, merged))

        for idx_entry in affected:
            if pair not in word_pair_counts[idx_entry]:
                continue
            old_symbols = entries[idx_entry]
            old_len = len(old_symbols)
            f = freqs[idx_entry]
            lang_idx = entry_lang_indices[idx_entry]
            code = lang_codes[lang_idx]

            # Remove all old pair contributions for this word.
            for old_pair, c in word_pair_counts[idx_entry].items():
                pair_counts_by_lang[old_pair][lang_idx] -= c * f
                if idx_entry in pair_to_words.get(old_pair, set()):
                    pair_to_words[old_pair].discard(idx_entry)

            new_symbols = merge_symbols(old_symbols, pair, merged)
            new_len = len(new_symbols)
            entries[idx_entry] = new_symbols
            reduction = old_len - new_len
            if reduction > 0:
                current_tokens[code] -= reduction * f

            new_pc = pair_counter(new_symbols)
            word_pair_counts[idx_entry] = new_pc
            for new_pair, c in new_pc.items():
                pair_counts_by_lang[new_pair][lang_idx] += c * f
                pair_to_words[new_pair].add(idx_entry)

        steps_since_rebuild += 1
        merge_count = len(merges)
        if merge_count == 1 or merge_count % checkpoint_every == 0 or len(vocab) == vocab_size:
            xs = x_by_code()
            mx, mn, sc = score_from_x(xs)
            progress.append({
                "merge_count": merge_count,
                "vocab_size_so_far": len(vocab),
                "x_by_language": {LANG_NAME_BY_CODE[c]: float(xs[c]) for c in lang_codes},
                "english_constraint_pass": bool(xs["en"] <= english_max_x),
                "max_x": float(mx),
                "min_x": float(mn),
                "delta_x": float(mx - mn),
                "score": sc,
            })

    unused_id = 0
    while len(vocab) < vocab_size:
        tok = f"<unused_{unused_id:05d}>"
        unused_id += 1
        if tok not in seen_vocab:
            vocab.append(tok)
            seen_vocab.add(tok)

    model = BPEModel(specials=list(SPECIAL_TOKENS), vocab=vocab, merges=merges, initial_units=initial_units)
    final_x = x_by_code()
    max_x, min_x, final_score = score_from_x(final_x)
    diagnostics = {
        "strategy": "grapheme_cluster_constrained_greedy_bpe_v3",
        "config_name": config_name,
        "english_max_x_constraint": english_max_x,
        "english_constraint_pass": bool(final_x["en"] <= english_max_x),
        "learned_token_budget": vocab_size - len(SPECIAL_TOKENS),
        "base_symbol_count": len(vocab) - len(SPECIAL_TOKENS) - len(merges) - len([t for t in vocab if t.startswith("<unused_")]),
        "merge_count": len(merges),
        "hyperparameters": {
            "english_boost": english_boost,
            "fairness_strength": fairness_strength,
            "worst_boost": worst_boost,
            "min_pressure": min_pressure,
            "rebuild_every": rebuild_every,
            "constraint_start_merge": constraint_start_merge,
        },
        "selected_x_by_language": {LANG_NAME_BY_CODE[c]: float(final_x[c]) for c in lang_codes},
        "selected_english_constraint_pass": bool(final_x["en"] <= english_max_x),
        "selected_delta_x": float(max_x - min_x),
        "selected_score": final_score,
        "progress_checkpoints": progress,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "explanation": "At each BPE step the trainer scores candidate merges by estimated reduction in X, with dynamic pressure for English while X>1.2 and for languages currently far above the minimum X.",
    }
    return model, diagnostics


def run_constrained_greedy_search(
    corpus: Dict[str, str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    initial_units: str = INITIAL_UNITS,
    english_max_x: float = 1.20,
) -> Tuple[BPEModel, Dict[str, object]]:
    """Try a small set of objective-guided greedy configurations and choose the best valid one."""
    configs = [
        # Original three retained for comparison
        {"config_name": "balanced_constraint", "english_boost": 18.0, "fairness_strength": 5.5, "worst_boost": 3.0, "rebuild_every": 18, "constraint_start_merge": 0},
        {"config_name": "fairness_heavy", "english_boost": 24.0, "fairness_strength": 7.0, "worst_boost": 4.0, "rebuild_every": 18, "constraint_start_merge": 0},
        {"config_name": "english_constraint_strong", "english_boost": 30.0, "fairness_strength": 4.0, "worst_boost": 2.5, "rebuild_every": 16, "constraint_start_merge": 0},

        # New v2 candidates: keep high fairness but raise English pressure moderately.
        {"config_name": "moderate_english_26", "english_boost": 26.0, "fairness_strength": 6.5, "worst_boost": 4.0, "rebuild_every": 16, "constraint_start_merge": 0},
        {"config_name": "moderate_english_28", "english_boost": 28.0, "fairness_strength": 6.5, "worst_boost": 4.0, "rebuild_every": 16, "constraint_start_merge": 0},
        {"config_name": "moderate_english_30_fair", "english_boost": 30.0, "fairness_strength": 6.5, "worst_boost": 4.0, "rebuild_every": 16, "constraint_start_merge": 0},

        # New v2 candidates: fair-first, then English rescue late.
        # These are intended to preserve the low delta of fairness_heavy, while forcing English under 1.2 near the end.
        {"config_name": "fair_first_rescue_4500", "english_boost": 38.0, "fairness_strength": 7.0, "worst_boost": 4.0, "rebuild_every": 14, "constraint_start_merge": 4500},
        {"config_name": "fair_first_rescue_5500", "english_boost": 42.0, "fairness_strength": 7.0, "worst_boost": 4.0, "rebuild_every": 14, "constraint_start_merge": 5500},
        {"config_name": "fair_first_rescue_6500", "english_boost": 48.0, "fairness_strength": 7.0, "worst_boost": 4.0, "rebuild_every": 14, "constraint_start_merge": 6500},
        {"config_name": "fair_first_rescue_7000", "english_boost": 58.0, "fairness_strength": 7.0, "worst_boost": 4.0, "rebuild_every": 12, "constraint_start_merge": 7000},

        # v3 targeted-rescue candidates. v2 showed that fair-first candidates had excellent delta
        # but English stayed around 1.26-1.39. These start the English rescue earlier and use
        # much stronger pressure, while preserving a fair prefix.
        {"config_name": "targeted_rescue_3000_b60", "english_boost": 60.0, "fairness_strength": 6.8, "worst_boost": 3.8, "rebuild_every": 12, "constraint_start_merge": 3000},
        {"config_name": "targeted_rescue_3200_b70", "english_boost": 70.0, "fairness_strength": 6.8, "worst_boost": 3.8, "rebuild_every": 12, "constraint_start_merge": 3200},
        {"config_name": "targeted_rescue_3500_b70", "english_boost": 70.0, "fairness_strength": 6.5, "worst_boost": 3.6, "rebuild_every": 12, "constraint_start_merge": 3500},
        {"config_name": "targeted_rescue_3500_b90", "english_boost": 90.0, "fairness_strength": 6.5, "worst_boost": 3.6, "rebuild_every": 12, "constraint_start_merge": 3500},
        {"config_name": "targeted_rescue_3800_b90", "english_boost": 90.0, "fairness_strength": 6.3, "worst_boost": 3.5, "rebuild_every": 12, "constraint_start_merge": 3800},
        {"config_name": "targeted_rescue_4000_b100", "english_boost": 100.0, "fairness_strength": 6.0, "worst_boost": 3.2, "rebuild_every": 12, "constraint_start_merge": 4000},
        {"config_name": "targeted_rescue_4200_b120", "english_boost": 120.0, "fairness_strength": 5.8, "worst_boost": 3.0, "rebuild_every": 10, "constraint_start_merge": 4200},
        {"config_name": "targeted_rescue_4500_b140", "english_boost": 140.0, "fairness_strength": 5.5, "worst_boost": 2.8, "rebuild_every": 10, "constraint_start_merge": 4500},

        # Very strong rescue fallback: if these pass, they should often beat the original
        # english_constraint_strong because they do not spend the earliest merges on English.
        {"config_name": "strong_late_rescue_2500_b90", "english_boost": 90.0, "fairness_strength": 5.8, "worst_boost": 3.0, "rebuild_every": 10, "constraint_start_merge": 2500},
        {"config_name": "strong_late_rescue_3000_b110", "english_boost": 110.0, "fairness_strength": 5.5, "worst_boost": 2.8, "rebuild_every": 10, "constraint_start_merge": 3000},
        {"config_name": "strong_late_rescue_3500_b130", "english_boost": 130.0, "fairness_strength": 5.2, "worst_boost": 2.6, "rebuild_every": 10, "constraint_start_merge": 3500},
    ]
    results: List[Tuple[BPEModel, Dict[str, object]]] = []
    print("Training objective-guided constrained BPE v3 candidates...", file=sys.stderr)
    for cfg in configs:
        print(f"  Candidate: {cfg['config_name']}", file=sys.stderr)
        model, diag = train_objective_guided_bpe(
            corpus_by_code=corpus,
            vocab_size=vocab_size,
            initial_units=initial_units,
            english_max_x=english_max_x,
            english_boost=float(cfg["english_boost"]),
            fairness_strength=float(cfg["fairness_strength"]),
            worst_boost=float(cfg["worst_boost"]),
            rebuild_every=int(cfg["rebuild_every"]),
            constraint_start_merge=int(cfg.get("constraint_start_merge", 0)),
            config_name=str(cfg["config_name"]),
        )
        results.append((model, diag))
        xs_print = diag.get("selected_x_by_language", {})
        print(
            f"    English pass={diag['english_constraint_pass']} en={float(xs_print.get('English', 0.0)):.4f} "
            f"delta={diag['selected_delta_x']:.4f} score={diag['selected_score']}",
            file=sys.stderr,
        )

    def rank_key(item: Tuple[BPEModel, Dict[str, object]]):
        diag = item[1]
        # Prefer passing English; among passing candidates choose smallest delta. If none pass,
        # choose the candidate with lowest English X, then smallest delta.
        xs = diag.get("selected_x_by_language", {})
        en_x = float(xs.get("English", 999.0))
        passes = bool(diag.get("english_constraint_pass"))
        return (not passes, float(diag.get("selected_delta_x", 999.0)) if passes else en_x, float(diag.get("selected_delta_x", 999.0)))

    best_model, best_diag = sorted(results, key=rank_key)[0]
    leaderboard = []
    for _, diag in sorted(results, key=rank_key):
        leaderboard.append({
            "config_name": diag.get("config_name"),
            "english_constraint_pass": bool(diag.get("english_constraint_pass")),
            "x_by_language": diag.get("selected_x_by_language", {}),
            "delta_x": diag.get("selected_delta_x"),
            "score": diag.get("selected_score"),
            "hyperparameters": diag.get("hyperparameters", {}),
            "elapsed_seconds": diag.get("elapsed_seconds"),
        })
    best_diag = dict(best_diag)
    best_diag["candidate_leaderboard"] = leaderboard
    best_diag["searched_candidate_count"] = len(results)
    best_diag["exact_checked_count"] = len(results)
    return best_model, best_diag

def non_special_vocab(model: BPEModel) -> List[str]:
    return model.vocab[len(SPECIAL_TOKENS):]


def base_symbol_count_for_text(text: str, initial_units: str = INITIAL_UNITS) -> int:
    return len({s for w in word_units(text) for s in initial_symbols_for_word(w, mode=initial_units)})


def truncate_model(model: BPEModel, learned_budget: int) -> BPEModel:
    """Keep the first `learned_budget` non-special tokens from a trained BPE model.

    Because train_bpe appends all base symbols before merge tokens, this produces a valid
    lower-budget BPE prefix as long as learned_budget is at least the base-symbol count.
    """
    learned = non_special_vocab(model)[:max(0, learned_budget)]
    vocab = list(SPECIAL_TOKENS) + list(learned)
    included = set(vocab)
    merges = [(a, b, m) for (a, b, m) in model.merges if m in included]
    return BPEModel(specials=list(SPECIAL_TOKENS), vocab=vocab, merges=merges, initial_units=model.initial_units)


def combine_quota_models(
    lang_max_models: Dict[str, BPEModel],
    quotas: Dict[str, int],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> BPEModel:
    """Combine language-specific BPE prefixes into one shared 10k tokenizer.

    The languages mostly use different scripts, so their merge spaces are naturally separated.
    Shared punctuation/digit tokens are deduplicated. Remaining budget is filled with reserved
    unused tokens so the assignment contract is exactly 10,000 vocabulary entries including
    special tokens.
    """
    vocab: List[str] = list(SPECIAL_TOKENS)
    seen_vocab = set(vocab)
    merges: List[Tuple[str, str, str]] = []
    seen_pairs: set[Tuple[str, str]] = set()

    for lang in LANGUAGES:
        code = str(lang["code"])
        part = truncate_model(lang_max_models[code], int(quotas[code]))
        for tok in part.vocab[len(SPECIAL_TOKENS):]:
            if tok not in seen_vocab:
                if len(vocab) >= vocab_size:
                    raise ValueError("Combined quota vocabulary exceeds requested vocab size")
                vocab.append(tok)
                seen_vocab.add(tok)
        for a, b, merged in part.merges:
            # Only keep rules whose output token is in the shared vocabulary.
            if merged in seen_vocab and (a, b) not in seen_pairs:
                merges.append((a, b, merged))
                seen_pairs.add((a, b))

    unused_id = 0
    while len(vocab) < vocab_size:
        tok = f"<unused_{unused_id:05d}>"
        unused_id += 1
        if tok not in seen_vocab:
            vocab.append(tok)
            seen_vocab.add(tok)

    return BPEModel(specials=list(SPECIAL_TOKENS), vocab=vocab, merges=merges, initial_units=INITIAL_UNITS)


def fertility_for_model(model: BPEModel, text: str) -> Tuple[int, int, float]:
    words = word_units(text)
    n_words = len(words)
    n_tokens = len(model.encode_text(text))
    x = (n_tokens / n_words) if n_words else 0.0
    return n_tokens, n_words, x


def make_range(lo: int, hi: int, step: int) -> List[int]:
    lo = int(lo); hi = int(hi); step = max(1, int(step))
    if hi < lo:
        return []
    vals = list(range(lo, hi + 1, step))
    if vals and vals[-1] != hi:
        vals.append(hi)
    if not vals:
        vals = [lo]
    return sorted(set(vals))


def score_from_x(x_by_code: Dict[str, float]) -> Tuple[float, float, float | None]:
    vals = list(x_by_code.values())
    mx, mn = max(vals), min(vals)
    d = mx - mn
    return mx, mn, (1000.0 / d if d > 0 else None)


def run_quota_search(
    corpus: Dict[str, str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    initial_units: str = INITIAL_UNITS,
    english_max_x: float = 1.20,
    coarse_step: int = 250,
    refine_step: int = 50,
    top_k_exact: int = 80,
) -> Tuple[BPEModel, Dict[str, object]]:
    """Search language token quotas and return the best exact shared tokenizer.

    Search objective: enforce English X <= 1.2, then minimize max(X)-min(X). The search
    first estimates fertility using per-language BPE prefixes, then verifies the best
    candidates with the exact combined shared tokenizer.
    """
    learned_total = vocab_size - len(SPECIAL_TOKENS)

    # Train high-capacity per-language models once. Quota candidates are cheap prefixes.
    print("Training per-language max BPE models for quota search...", file=sys.stderr)
    lang_max_models: Dict[str, BPEModel] = {}
    base_counts: Dict[str, int] = {}
    available_counts: Dict[str, int] = {}
    for lang in LANGUAGES:
        code = str(lang["code"])
        lang_max_models[code] = train_bpe([corpus[code]], vocab_size=vocab_size, initial_units=initial_units)
        base_counts[code] = base_symbol_count_for_text(corpus[code], initial_units=initial_units)
        available_counts[code] = len(non_special_vocab(lang_max_models[code]))
        print(f"  {lang['name']}: base={base_counts[code]}, available={available_counts[code]}", file=sys.stderr)

    # Upper caps keep the search practical but still allow English enough budget to satisfy <=1.2.
    caps = {
        "en": min(7000, available_counts["en"], learned_total),
        "hi": min(4500, available_counts["hi"], learned_total),
        "te": min(5500, available_counts["te"], learned_total),
        "kn": min(4500, available_counts["kn"], learned_total),
    }
    lowers = {c: max(base_counts[c], 1) for c in base_counts}
    # Ensure at least the lower bound is feasible for all languages.
    if sum(lowers.values()) > learned_total:
        raise RuntimeError(f"Base symbols alone exceed learned token budget: {sum(lowers.values())} > {learned_total}")

    x_cache: Dict[Tuple[str, int], Tuple[int, int, float]] = {}
    def get_x(code: str, q: int) -> Tuple[int, int, float]:
        key = (code, int(q))
        if key not in x_cache:
            tm = truncate_model(lang_max_models[code], int(q))
            x_cache[key] = fertility_for_model(tm, corpus[code])
        return x_cache[key]

    def candidate_record(quotas: Dict[str, int], stage: str) -> Dict[str, object] | None:
        if sum(quotas.values()) != learned_total:
            return None
        for code, q in quotas.items():
            if q < lowers[code] or q > caps[code]:
                return None
        x_by_code = {code: get_x(code, q)[2] for code, q in quotas.items()}
        max_x, min_x, approx_score = score_from_x(x_by_code)
        delta = max_x - min_x
        en_pass = x_by_code["en"] <= english_max_x
        return {
            "stage": stage,
            "quotas": dict(quotas),
            "x_by_code": x_by_code,
            "english_constraint_pass": en_pass,
            "max_x": max_x,
            "min_x": min_x,
            "delta_x": delta,
            "score": approx_score,
        }

    all_candidates: List[Dict[str, object]] = []

    def add_grid(en_vals: List[int], hi_vals: List[int], te_vals: List[int], stage: str):
        seen_local: set[Tuple[int, int, int, int]] = set()
        for en_q in en_vals:
            for hi_q in hi_vals:
                for te_q in te_vals:
                    kn_q = learned_total - en_q - hi_q - te_q
                    key = (en_q, hi_q, te_q, kn_q)
                    if key in seen_local:
                        continue
                    seen_local.add(key)
                    rec = candidate_record({"en": en_q, "hi": hi_q, "te": te_q, "kn": kn_q}, stage)
                    if rec is not None:
                        all_candidates.append(rec)

    en_vals = make_range(lowers["en"], caps["en"], coarse_step)
    hi_vals = make_range(lowers["hi"], caps["hi"], coarse_step)
    te_vals = make_range(lowers["te"], caps["te"], coarse_step)
    add_grid(en_vals, hi_vals, te_vals, "coarse")
    pass_candidates = [c for c in all_candidates if c["english_constraint_pass"]]

    # If coarse search cannot satisfy English, force a wider English-heavy sweep.
    if not pass_candidates:
        print("No coarse candidate met English <= 1.2; running English-heavy coarse sweep...", file=sys.stderr)
        en_vals = make_range(max(lowers["en"], 4500), caps["en"], max(100, coarse_step // 2))
        hi_vals = make_range(lowers["hi"], min(caps["hi"], 2500), coarse_step)
        te_vals = make_range(lowers["te"], caps["te"], coarse_step)
        add_grid(en_vals, hi_vals, te_vals, "english_heavy_coarse")
        pass_candidates = [c for c in all_candidates if c["english_constraint_pass"]]

    seed_pool = pass_candidates if pass_candidates else all_candidates
    seed_pool = sorted(seed_pool, key=lambda c: (float(c["delta_x"]), float(c["max_x"]), abs(float(c["x_by_code"]["en"]) - english_max_x)))[:20]

    # Refine around the best coarse candidates.
    for seed in seed_pool:
        q = seed["quotas"]
        radius = max(coarse_step, 300)
        en_vals = make_range(max(lowers["en"], q["en"] - radius), min(caps["en"], q["en"] + radius), refine_step)
        hi_vals = make_range(max(lowers["hi"], q["hi"] - radius), min(caps["hi"], q["hi"] + radius), refine_step)
        te_vals = make_range(max(lowers["te"], q["te"] - radius), min(caps["te"], q["te"] + radius), refine_step)
        add_grid(en_vals, hi_vals, te_vals, "refine")

    # Deduplicate approximate candidates.
    unique: Dict[Tuple[int, int, int, int], Dict[str, object]] = {}
    for c in all_candidates:
        q = c["quotas"]
        key = (q["en"], q["hi"], q["te"], q["kn"])
        old = unique.get(key)
        if old is None or float(c["delta_x"]) < float(old["delta_x"]):
            unique[key] = c
    all_candidates = list(unique.values())

    pass_candidates = [c for c in all_candidates if c["english_constraint_pass"]]
    approx_ranked = sorted(pass_candidates if pass_candidates else all_candidates,
                           key=lambda c: (float(c["delta_x"]), float(c["max_x"]), abs(float(c["x_by_code"]["en"]) - english_max_x)))

    # Verify top approximate candidates using the exact combined tokenizer.
    exact_records: List[Dict[str, object]] = []
    top = approx_ranked[:top_k_exact]
    print(f"Quota candidates: {len(all_candidates)} searched, {len(pass_candidates)} pass English approx. Exact-checking {len(top)}...", file=sys.stderr)
    for cand in top:
        quotas = cand["quotas"]
        model = combine_quota_models(lang_max_models, quotas, vocab_size=vocab_size)
        x_by_code = {code: fertility_for_model(model, corpus[code])[2] for code in [str(l["code"]) for l in LANGUAGES]}
        mx, mn, sc = score_from_x(x_by_code)
        exact_records.append({
            "quotas": dict(quotas),
            "x_by_code": x_by_code,
            "english_constraint_pass": x_by_code["en"] <= english_max_x,
            "max_x": mx,
            "min_x": mn,
            "delta_x": mx - mn,
            "score": sc,
            "approx_delta_x": cand["delta_x"],
            "approx_x_by_code": cand["x_by_code"],
        })

    exact_pass = [r for r in exact_records if r["english_constraint_pass"]]
    final_record = sorted(exact_pass if exact_pass else exact_records,
                          key=lambda c: (float(c["delta_x"]), float(c["max_x"]), abs(float(c["x_by_code"]["en"]) - english_max_x)))[0]
    final_model = combine_quota_models(lang_max_models, final_record["quotas"], vocab_size=vocab_size)

    # Friendly labels and compact leaderboard for the widget.
    lang_names = {str(l["code"]): str(l["name"]) for l in LANGUAGES}
    final_quotas_named = {lang_names[c]: int(q) for c, q in final_record["quotas"].items()}
    final_x_named = {lang_names[c]: float(x) for c, x in final_record["x_by_code"].items()}
    top_exact = []
    for r in sorted(exact_records, key=lambda c: (not c["english_constraint_pass"], float(c["delta_x"]), float(c["max_x"])))[:10]:
        top_exact.append({
            "quotas": {lang_names[c]: int(q) for c, q in r["quotas"].items()},
            "x_by_language": {lang_names[c]: float(x) for c, x in r["x_by_code"].items()},
            "english_constraint_pass": bool(r["english_constraint_pass"]),
            "delta_x": float(r["delta_x"]),
            "score": r["score"],
        })

    diagnostics = {
        "strategy": "grapheme_cluster_language_quota_search",
        "english_max_x_constraint": english_max_x,
        "learned_token_budget": learned_total,
        "base_symbol_counts": {lang_names[c]: int(v) for c, v in base_counts.items()},
        "quota_caps": {lang_names[c]: int(v) for c, v in caps.items()},
        "searched_candidate_count": len(all_candidates),
        "approx_english_pass_count": len(pass_candidates),
        "exact_checked_count": len(exact_records),
        "selected_quotas": final_quotas_named,
        "selected_x_by_language": final_x_named,
        "selected_english_constraint_pass": bool(final_record["english_constraint_pass"]),
        "selected_delta_x": float(final_record["delta_x"]),
        "selected_score": final_record["score"],
        "top_exact_candidates": top_exact,
    }
    return final_model, diagnostics


def token_visible(token: str) -> str:
    return token.replace(" ", "␠").replace("\t", "⇥").replace("\n", "↵")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def language_stats(lang: Dict[str, object], text: str, model: BPEModel, source_status: Dict[str, object]) -> Dict[str, object]:
    words = word_units(text)
    encoded = model.encode_text(text)
    decoded = model.decode_tokens(encoded)
    faithful_units = faithful_unit_count(text)
    char_tokens = initial_baseline_tokens(text, mode=model.initial_units)
    return {
        "code": lang["code"],
        "language": lang["name"],
        "wiki_title": lang["title"],
        "source_url": lang["url"],
        "word_units": len(words),
        "unique_word_types": len(set(words)),
        "bpe_tokens": len(encoded),
        "unique_bpe_tokens_used": len(set(encoded)),
        "characters": len(text),
        "initial_baseline_tokens": char_tokens,
        "initial_baseline_x": (char_tokens / len(words)) if words else 0.0,
        "char_baseline_tokens": char_tokens,
        "char_baseline_x": (char_tokens / len(words)) if words else 0.0,
        "fertility_x": (len(encoded) / len(words)) if words else 0.0,
        "fertility_x_rounded": round((len(encoded) / len(words)) if words else 0.0, 4),
        "faithful_units": faithful_units,
        "faithful_fertility_x": (len(encoded) / faithful_units) if faithful_units else 0.0,
        "faithful_fertility_x_rounded": round((len(encoded) / faithful_units) if faithful_units else 0.0, 4),
        "round_trip_visible_valid": visible_non_whitespace(decoded) == visible_non_whitespace(text),
        "round_trip_exact_valid": decoded == text,
        "source_status": source_status,
    }


def compute_score(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], float, float, float, float | None, str]:
    sorted_rows = sorted(rows, key=lambda x: float(x["fertility_x"]), reverse=True)
    max_x = max(float(x["fertility_x"]) for x in rows)
    min_x = min(float(x["fertility_x"]) for x in rows)
    delta = max_x - min_x
    score = (1000.0 / delta) if delta > 0 else None
    score_display = f"{score:,.2f}" if score is not None else "∞"
    return sorted_rows, max_x, min_x, delta, score, score_display


def compute_score_for_field(rows: Sequence[Dict[str, object]], field: str) -> Dict[str, object]:
    sorted_rows = sorted(rows, key=lambda row: float(row[field]), reverse=True)
    max_x = max(float(row[field]) for row in rows)
    min_x = min(float(row[field]) for row in rows)
    delta = max_x - min_x
    score = (1000.0 / delta) if delta > 0 else None
    return {
        "max_x": max_x,
        "min_x": min_x,
        "delta_x": delta,
        "score": score,
        "score_display": f"{score:,.2f}" if score is not None else "∞",
        "sorted_by_x_desc": [str(row["language"]) for row in sorted_rows],
    }




# ---------------------------------------------------------------------------
# v4 performance layer: integer-ID tuples + deduplicated word-frequency corpus
# ---------------------------------------------------------------------------
# The earlier trainers already used Counter(word_units(...)) in several places, but the
# working representation for words was still tuple[str, ...].  For bigger candidate
# searches this creates a lot of repeated string comparisons and string tuple hashing.
# The definitions below intentionally override train_bpe(...) and
# train_objective_guided_bpe(...) above.  They preserve the same external tokenizer.json
# format, but train internally with:
#   - one deduplicated word-frequency dictionary per corpus/language
#   - words represented as tuple[int, ...]
#   - BPE pairs represented as tuple[int, int]
# This is a speed/memory optimization only; the learned merge rules are still exported as
# readable strings and the fertility numbers remain computed by the same BPE model.

class IntSymbolTable:
    """Bidirectional mapping between token strings and compact integer symbol IDs."""

    def __init__(self) -> None:
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: List[str] = []

    def get_id(self, text: str) -> int:
        found = self.str_to_id.get(text)
        if found is not None:
            return found
        idx = len(self.id_to_str)
        self.str_to_id[text] = idx
        self.id_to_str.append(text)
        return idx

    def text(self, idx: int) -> str:
        return self.id_to_str[idx]

    def __contains__(self, text: str) -> bool:  # pragma: no cover - convenience
        return text in self.str_to_id


def initial_symbol_ids_for_word(word: str, symtab: IntSymbolTable, mode: str = INITIAL_UNITS) -> Tuple[int, ...]:
    return tuple(symtab.get_id(s) for s in initial_symbols_for_word(word, mode=mode))


def merge_symbol_ids(symbols: Sequence[int], pair: Tuple[int, int], merged_id: int) -> Tuple[int, ...]:
    out: List[int] = []
    i = 0
    n = len(symbols)
    a, b = pair
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def pair_counter_ids(symbols: Sequence[int]) -> Counter[Tuple[int, int]]:
    c: Counter[Tuple[int, int]] = collections.Counter()
    for i in range(len(symbols) - 1):
        c[(symbols[i], symbols[i + 1])] += 1
    return c


def add_base_symbols_to_int_vocab(
    words: Iterable[str],
    symtab: IntSymbolTable,
    vocab: List[str],
    seen_vocab: set[str],
    vocab_size: int,
    initial_units: str,
) -> None:
    """Add all initial symbols once, in stable string-sorted order."""
    base_symbols = set()
    for word in words:
        base_symbols.update(initial_symbols_for_word(word, mode=initial_units))
    needed = len([s for s in base_symbols if s not in seen_vocab])
    if len(vocab) + needed > vocab_size:
        raise RuntimeError(
            f"Initial symbols alone exceed vocab budget: base_needed={needed}, "
            f"current_vocab={len(vocab)}, vocab_size={vocab_size}"
        )
    for sym in sorted(base_symbols):
        if sym not in seen_vocab:
            symtab.get_id(sym)
            vocab.append(sym)
            seen_vocab.add(sym)


def make_merged_symbol_id(
    pair: Tuple[int, int],
    symtab: IntSymbolTable,
    used_merge_names: set[str],
) -> Tuple[int, str, str, str]:
    """Create a new integer symbol and readable merged-token string for a pair."""
    left = symtab.text(pair[0])
    right = symtab.text(pair[1])
    merged = left + right
    if merged in used_merge_names:
        suffix = 2
        candidate = f"{merged}⟨{suffix}⟩"
        while candidate in used_merge_names:
            suffix += 1
            candidate = f"{merged}⟨{suffix}⟩"
        merged = candidate
    merged_id = symtab.get_id(merged)
    return merged_id, left, right, merged


def train_bpe(texts: Sequence[str], vocab_size: int = DEFAULT_VOCAB_SIZE, initial_units: str = INITIAL_UNITS) -> BPEModel:  # type: ignore[override]
    """Standard BPE trainer using deduplicated word-frequency entries and integer symbol IDs."""
    word_counts: Counter[str] = collections.Counter()
    for text in texts:
        word_counts.update(word_units(text))

    symtab = IntSymbolTable()
    vocab: List[str] = list(SPECIAL_TOKENS)
    seen_vocab: set[str] = set(vocab)
    add_base_symbols_to_int_vocab(word_counts.keys(), symtab, vocab, seen_vocab, vocab_size, initial_units)

    # One entry per unique whitespace word. Frequency preserves the original corpus weighting.
    entries: List[Tuple[int, ...]] = [initial_symbol_ids_for_word(w, symtab, mode=initial_units) for w in word_counts.keys()]
    freqs: List[int] = list(word_counts.values())

    pair_counts: Counter[Tuple[int, int]] = collections.Counter()
    pair_to_words: Dict[Tuple[int, int], set[int]] = collections.defaultdict(set)
    word_pair_counts: List[Counter[Tuple[int, int]]] = []

    for idx, symbols in enumerate(entries):
        pc = pair_counter_ids(symbols)
        word_pair_counts.append(pc)
        f = freqs[idx]
        for pair, count in pc.items():
            pair_counts[pair] += count * f
            pair_to_words[pair].add(idx)

    heap: List[Tuple[int, Tuple[int, int]]] = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges: List[Tuple[str, str, str]] = []
    used_merge_names = set(seen_vocab)

    while len(vocab) < vocab_size and heap:
        neg_count, pair = heapq.heappop(heap)
        current_count = pair_counts.get(pair, 0)
        if current_count <= 0 or -neg_count != current_count:
            continue

        affected = list(pair_to_words.get(pair, set()))
        if not affected:
            pair_counts[pair] = 0
            continue

        merged_id, left, right, merged = make_merged_symbol_id(pair, symtab, used_merge_names)
        vocab.append(merged)
        seen_vocab.add(merged)
        used_merge_names.add(merged)
        merges.append((left, right, merged))

        touched_pairs: set[Tuple[int, int]] = set()
        for idx in affected:
            if pair not in word_pair_counts[idx]:
                continue
            f = freqs[idx]
            # Remove all old pair contributions for this unique word.
            for old_pair, c in word_pair_counts[idx].items():
                pair_counts[old_pair] -= c * f
                touched_pairs.add(old_pair)
                if idx in pair_to_words.get(old_pair, set()):
                    pair_to_words[old_pair].discard(idx)

            new_symbols = merge_symbol_ids(entries[idx], pair, merged_id)
            entries[idx] = new_symbols
            new_pc = pair_counter_ids(new_symbols)
            word_pair_counts[idx] = new_pc
            for new_pair, c in new_pc.items():
                pair_counts[new_pair] += c * f
                pair_to_words[new_pair].add(idx)
                touched_pairs.add(new_pair)

        for p in touched_pairs:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))

    unused_id = 0
    while len(vocab) < vocab_size:
        tok = f"<unused_{unused_id:05d}>"
        unused_id += 1
        if tok not in seen_vocab:
            vocab.append(tok)
            seen_vocab.add(tok)

    return BPEModel(specials=list(SPECIAL_TOKENS), vocab=vocab, merges=merges, initial_units=initial_units)


def train_objective_guided_bpe(
    corpus_by_code: Dict[str, str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    initial_units: str = INITIAL_UNITS,
    english_max_x: float = 1.20,
    english_boost: float = 22.0,
    fairness_strength: float = 5.0,
    worst_boost: float = 3.0,
    min_pressure: float = 0.02,
    rebuild_every: int = 18,
    checkpoint_every: int = 500,
    constraint_start_merge: int = 0,
    config_name: str = "objective_guided",
) -> Tuple[BPEModel, Dict[str, object]]:  # type: ignore[override]
    """Objective-guided BPE using integer symbol IDs and per-language word-frequency dictionaries."""
    lang_codes = list(LANG_CODE_ORDER)
    n_langs = len(lang_codes)
    word_unit_counts = {code: len(word_units(corpus_by_code[code])) for code in lang_codes}

    # Build one word-frequency dictionary per language. This is the corpus seen by BPE:
    # unique word -> frequency. Repeated word occurrences do not create repeated entries.
    word_counts_by_code: Dict[str, Counter[str]] = {
        code: collections.Counter(word_units(corpus_by_code[code])) for code in lang_codes
    }

    symtab = IntSymbolTable()
    vocab: List[str] = list(SPECIAL_TOKENS)
    seen_vocab: set[str] = set(vocab)
    all_unique_words: List[str] = []
    for code in lang_codes:
        all_unique_words.extend(word_counts_by_code[code].keys())
    add_base_symbols_to_int_vocab(all_unique_words, symtab, vocab, seen_vocab, vocab_size, initial_units)

    entries: List[Tuple[int, ...]] = []
    freqs: List[int] = []
    entry_lang_indices: List[int] = []
    unique_word_entries_by_language: Dict[str, int] = {}
    for code in lang_codes:
        unique_word_entries_by_language[LANG_NAME_BY_CODE[code]] = len(word_counts_by_code[code])
        lang_idx = LANG_INDEX_BY_CODE[code]
        for word, freq in word_counts_by_code[code].items():
            entries.append(initial_symbol_ids_for_word(word, symtab, mode=initial_units))
            freqs.append(int(freq))
            entry_lang_indices.append(lang_idx)

    current_tokens = {code: 0 for code in lang_codes}
    for symbols, freq, lang_idx in zip(entries, freqs, entry_lang_indices):
        current_tokens[lang_codes[lang_idx]] += len(symbols) * freq

    def zero_counts() -> List[int]:
        return [0 for _ in range(n_langs)]

    pair_counts_by_lang: Dict[Tuple[int, int], List[int]] = collections.defaultdict(zero_counts)
    pair_to_words: Dict[Tuple[int, int], set[int]] = collections.defaultdict(set)
    word_pair_counts: List[Counter[Tuple[int, int]]] = []

    for idx_entry, symbols in enumerate(entries):
        pc = pair_counter_ids(symbols)
        word_pair_counts.append(pc)
        f = freqs[idx_entry]
        lang_idx = entry_lang_indices[idx_entry]
        for pair, count in pc.items():
            pair_counts_by_lang[pair][lang_idx] += count * f
            pair_to_words[pair].add(idx_entry)

    def x_by_code() -> Dict[str, float]:
        return {
            code: (current_tokens[code] / word_unit_counts[code] if word_unit_counts[code] else 0.0)
            for code in lang_codes
        }

    def compute_pressures(merge_count_so_far: int) -> List[float]:
        xs = x_by_code()
        max_x = max(xs.values())
        min_x = min(xs.values())
        delta = max(max_x - min_x, 1e-9)
        pressures: List[float] = []
        for code in lang_codes:
            x = xs[code]
            normalized_gap = max(0.0, (x - min_x) / delta)
            pressure = min_pressure + fairness_strength * (normalized_gap ** 1.35)
            if x >= max_x - 1e-12:
                pressure += worst_boost

            if code == "en":
                if x > english_max_x and merge_count_so_far >= constraint_start_merge:
                    over = max(0.0, x - english_max_x)
                    pressure += english_boost * (1.0 + min(2.5, over / 0.20))
                elif x <= min_x + 0.03:
                    pressure *= 0.12
            else:
                if x <= min_x + 0.04:
                    pressure *= 0.18
            pressures.append(max(min_pressure, pressure))
        return pressures

    def pair_score(pair: Tuple[int, int], pressures: Sequence[float]) -> float:
        counts = pair_counts_by_lang.get(pair)
        if not counts:
            return 0.0
        score = 0.0
        total = 0
        for i, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            total += cnt
            code = lang_codes[i]
            denom = max(1, word_unit_counts[code])
            score += float(pressures[i]) * (cnt / denom)
        if total <= 0:
            return 0.0
        return score + 1e-9 * math.log1p(total)

    def rebuild_heap(merge_count_so_far: int) -> Tuple[List[Tuple[float, int, Tuple[int, int]]], List[float]]:
        pressures = compute_pressures(merge_count_so_far)
        heap: List[Tuple[float, int, Tuple[int, int]]] = []
        for pair, counts in pair_counts_by_lang.items():
            total = sum(counts)
            if total <= 0:
                continue
            sc = pair_score(pair, pressures)
            if sc > 0:
                heap.append((-sc, -total, pair))
        heapq.heapify(heap)
        return heap, pressures

    merges: List[Tuple[str, str, str]] = []
    used_merge_names = set(seen_vocab)
    progress: List[Dict[str, object]] = []
    heap, pressures = rebuild_heap(0)
    steps_since_rebuild = 0
    start_time = time.time()

    while len(vocab) < vocab_size:
        if not heap or steps_since_rebuild >= rebuild_every:
            heap, pressures = rebuild_heap(len(merges))
            steps_since_rebuild = 0
            if not heap:
                break

        _, _, pair = heapq.heappop(heap)
        counts_now = pair_counts_by_lang.get(pair)
        if not counts_now or sum(counts_now) <= 0:
            continue
        affected = list(pair_to_words.get(pair, set()))
        if not affected:
            pair_counts_by_lang[pair] = zero_counts()
            continue

        merged_id, left, right, merged = make_merged_symbol_id(pair, symtab, used_merge_names)
        vocab.append(merged)
        seen_vocab.add(merged)
        used_merge_names.add(merged)
        merges.append((left, right, merged))

        for idx_entry in affected:
            if pair not in word_pair_counts[idx_entry]:
                continue
            old_symbols = entries[idx_entry]
            old_len = len(old_symbols)
            f = freqs[idx_entry]
            lang_idx = entry_lang_indices[idx_entry]
            code = lang_codes[lang_idx]

            for old_pair, c in word_pair_counts[idx_entry].items():
                pair_counts_by_lang[old_pair][lang_idx] -= c * f
                if idx_entry in pair_to_words.get(old_pair, set()):
                    pair_to_words[old_pair].discard(idx_entry)

            new_symbols = merge_symbol_ids(old_symbols, pair, merged_id)
            new_len = len(new_symbols)
            entries[idx_entry] = new_symbols
            reduction = old_len - new_len
            if reduction > 0:
                current_tokens[code] -= reduction * f

            new_pc = pair_counter_ids(new_symbols)
            word_pair_counts[idx_entry] = new_pc
            for new_pair, c in new_pc.items():
                pair_counts_by_lang[new_pair][lang_idx] += c * f
                pair_to_words[new_pair].add(idx_entry)

        steps_since_rebuild += 1
        merge_count = len(merges)
        if merge_count == 1 or merge_count % checkpoint_every == 0 or len(vocab) == vocab_size:
            xs = x_by_code()
            mx, mn, sc = score_from_x(xs)
            progress.append({
                "merge_count": merge_count,
                "vocab_size_so_far": len(vocab),
                "x_by_language": {LANG_NAME_BY_CODE[c]: float(xs[c]) for c in lang_codes},
                "english_constraint_pass": bool(xs["en"] <= english_max_x),
                "max_x": float(mx),
                "min_x": float(mn),
                "delta_x": float(mx - mn),
                "score": sc,
            })

    unused_id = 0
    while len(vocab) < vocab_size:
        tok = f"<unused_{unused_id:05d}>"
        unused_id += 1
        if tok not in seen_vocab:
            vocab.append(tok)
            seen_vocab.add(tok)

    model = BPEModel(specials=list(SPECIAL_TOKENS), vocab=vocab, merges=merges, initial_units=initial_units)
    final_x = x_by_code()
    max_x, min_x, final_score = score_from_x(final_x)
    diagnostics = {
        "strategy": "grapheme_cluster_constrained_greedy_bpe_v4_int_id_wordfreq",
        "config_name": config_name,
        "english_max_x_constraint": english_max_x,
        "english_constraint_pass": bool(final_x["en"] <= english_max_x),
        "learned_token_budget": vocab_size - len(SPECIAL_TOKENS),
        "base_symbol_count": len(vocab) - len(SPECIAL_TOKENS) - len(merges) - len([t for t in vocab if t.startswith("<unused_")]),
        "merge_count": len(merges),
        "internal_training_representation": "tuple[int, ...] per unique word; pair keys are tuple[int, int]",
        "corpus_deduplication": "per-language Counter: unique whitespace word -> frequency",
        "unique_word_entries_by_language": unique_word_entries_by_language,
        "total_unique_word_entries": int(sum(unique_word_entries_by_language.values())),
        "hyperparameters": {
            "english_boost": english_boost,
            "fairness_strength": fairness_strength,
            "worst_boost": worst_boost,
            "min_pressure": min_pressure,
            "rebuild_every": rebuild_every,
            "constraint_start_merge": constraint_start_merge,
        },
        "selected_x_by_language": {LANG_NAME_BY_CODE[c]: float(final_x[c]) for c in lang_codes},
        "selected_english_constraint_pass": bool(final_x["en"] <= english_max_x),
        "selected_delta_x": float(max_x - min_x),
        "selected_score": final_score,
        "progress_checkpoints": progress,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "explanation": "Training uses integer ID tuples and per-language word-frequency dictionaries. Each legal BPE merge is still exported as readable string merge rules in tokenizer.json.",
    }
    return model, diagnostics


def build_data(
    output_dir: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    force_fallback: bool = False,
    strict_live: bool = True,
    train_fraction: float = 0.80,
) -> Dict[str, object]:
    """Fetch real Wikipedia pages, train BPE, and write widget data.

    Important: the normal/default path is live Wikipedia only. The embedded sample text is
    available only with --force-fallback for UI testing in offline environments. It is never
    used silently, because fallback numbers are not valid for the assignment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus: Dict[str, str] = {}
    source_status: Dict[str, Dict[str, object]] = {}
    build_mode = "live_wikipedia_full_articles"

    for lang in LANGUAGES:
        code = str(lang["code"])
        if force_fallback:
            text = clean_extract(FALLBACK_TEXT[code], lang["stop_headings"])
            status = {"source": "embedded_fallback_sample", "ok": False, "error": "forced sample mode; not valid for assignment"}
            build_mode = "offline_fallback_sample_not_for_submission"
        else:
            try:
                text = fetch_wikipedia_extract(lang)
                wc = len(word_units(text))
                if wc < 100:
                    raise RuntimeError(f"extract too short for {lang['name']}: {wc} word units")
                status = {"source": "wikipedia_api_full_plain_text", "ok": True, "error": None}
            except Exception as exc:
                raise RuntimeError(
                    f"Live Wikipedia fetch failed for {lang['name']} ({lang['url']}). "
                    "This build is intentionally strict so the assignment never reports sample/fallback numbers. "
                    "Run again on an internet-connected machine or deploy on Netlify. "
                    f"Original error: {exc}"
                ) from exc
        corpus[code] = text
        source_status[code] = status

    # Split is only for the sanity diagnostic. The submitted tokenizer below is trained on full pages.
    train_corpus: Dict[str, str] = {}
    eval_corpus: Dict[str, str] = {}
    split_stats: Dict[str, Dict[str, int]] = {}
    for lang in LANGUAGES:
        code = str(lang["code"])
        train_text, eval_text = split_for_raw_diagnostic(corpus[code], train_fraction=train_fraction)
        train_corpus[code] = train_text
        eval_corpus[code] = eval_text
        split_stats[code] = {
            "train_word_units": len(word_units(train_text)),
            "eval_word_units": len(word_units(eval_text)),
            "full_word_units": len(word_units(corpus[code])),
        }

    # Assignment tokenizer: grapheme-cluster BPE with objective-guided constrained greedy merge selection. Training internals use deduplicated word-frequency dictionaries and integer-ID symbol tuples.
    # The optimizer enforces English X <= 1.2 pressure during training and then minimizes max(X)-min(X).
    model, optimizer_diagnostics = run_constrained_greedy_search(corpus, vocab_size=vocab_size, initial_units=INITIAL_UNITS)

    token_freq_full: Counter[str] = collections.Counter()
    for lang in LANGUAGES:
        code = str(lang["code"])
        token_freq_full.update(model.encode_text(corpus[code]))

    per_language: List[Dict[str, object]] = []
    for lang in LANGUAGES:
        code = str(lang["code"])
        stats = language_stats(lang, corpus[code], model, source_status[code])
        stats.update(split_stats[code])
        stats["corpus_sha256"] = sha256_text(corpus[code])
        stats["initial_unit_mode"] = INITIAL_UNITS
        per_language.append(stats)

    sorted_langs, max_x, min_x, delta, score, score_display = compute_score(per_language)
    faithful_summary = compute_score_for_field(per_language, "faithful_fertility_x")
    visible_round_trip_valid = all(bool(row["round_trip_visible_valid"]) for row in per_language)
    exact_round_trip_valid = all(bool(row["round_trip_exact_valid"]) for row in per_language)

    # Held-out diagnostic: separate model trained on first 80%, evaluated on last 20%, to reveal memorisation risk.
    diagnostic_model = train_bpe([train_corpus[str(l["code"])] for l in LANGUAGES], vocab_size=vocab_size, initial_units=INITIAL_UNITS)
    heldout_per_language: List[Dict[str, object]] = []
    for lang in LANGUAGES:
        code = str(lang["code"])
        stats = language_stats(lang, eval_corpus[code], diagnostic_model, source_status[code])
        stats.update(split_stats[code])
        stats["corpus_sha256"] = sha256_text(eval_corpus[code])
        stats["initial_unit_mode"] = INITIAL_UNITS
        heldout_per_language.append(stats)
    held_sorted, held_max, held_min, held_delta, held_score, held_score_display = compute_score(heldout_per_language)

    vocab_records = []
    for i, tok in enumerate(model.vocab):
        vocab_records.append(
            {
                "id": i,
                "token": tok,
                "visible": token_visible(tok),
                "frequency": int(token_freq_full.get(tok, 0)),
                "frequency_train": int(token_freq_full.get(tok, 0)),
                "frequency_eval": int(token_freq_full.get(tok, 0)),
                "frequency_full": int(token_freq_full.get(tok, 0)),
                "is_special": tok in SPECIAL_TOKENS,
                "length_chars": len(tok),
            }
        )

    tokenizer_json = {
        "tokenizer_type": "word_boundary_bpe",
        "initial_unit_mode": INITIAL_UNITS,
        "segmentation_description": "Words are split on whitespace. Each word begins with ▁ and is then split into Unicode grapheme/akshara-like clusters before BPE merges are applied.",
        "decoding_description": "Recursively expand merge rules, concatenate symbols, and replace ▁ with a space. This preserves all non-whitespace characters; runs and kinds of whitespace are normalized by the original word-based design.",
        "faithfulness_contract": "decode(encode(text)) preserves the exact sequence of non-whitespace characters.",
        "description": "Shared grapheme-cluster BPE tokenizer trained on the full India Wikipedia articles in English, Hindi, Telugu, and Kannada, optimized with objective-guided constrained greedy BPE to enforce English X <= 1.2 and reduce cross-language fertility gap. Internally, training deduplicates each corpus into word-frequency dictionaries and represents words as integer ID tuples for faster candidate search.",
        "optimization": optimizer_diagnostics,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": len(model.vocab),
        "special_tokens": SPECIAL_TOKENS,
        "word_boundary_token": WORD_BOUNDARY,
        "vocab": vocab_records,
        "merges": [
            {"rank": i, "left": a, "right": b, "token": merged}
            for i, (a, b, merged) in enumerate(model.merges)
        ],
    }

    metrics_json = {
        "assignment": "India Wikipedia multilingual BPE tokenizer fertility fairness",
        "built_at_utc": now_iso(),
        "build_mode": build_mode,
        "evaluation_mode": "assignment_full_article_text",
        "validation": {
            "visible_non_whitespace_round_trip_valid": visible_round_trip_valid,
            "exact_whitespace_round_trip_valid": exact_round_trip_valid,
            "contract": "The decoded output must contain exactly the same non-whitespace characters in the same order. Whitespace normalization is reported separately.",
        },
        "method": {
            "corpus": "Live Wikipedia plain-text extracts for full India article body + headings; TextExtracts/plaintext removes tables and infoboxes, and the cleaner cuts reference/see-also/external-link sections by heading.",
            "languages": [
                {"code": l["code"], "language": l["name"], "wiki_title": l["title"], "source_url": l["url"]}
                for l in LANGUAGES
            ],
            "initial_units": "Unicode grapheme/akshara-like clusters, not raw Unicode codepoints",
            "training_weighting": "objective-guided constrained greedy BPE: candidate merges are scored by estimated reduction in fertility X, with dynamic pressure for English while X > 1.2 and for languages far above the minimum X. No text is deleted or sampled for the submitted score.",
            "optimization_strategy": "grapheme-cluster BPE + constrained greedy merge selection + integer-ID word-frequency training + candidate configuration search",
            "train_eval_split": "assignment metric uses full article text; held-out diagnostic separately trains on first 80% and evaluates on last 20%",
            "why_holdout": "The held-out diagnostic is shown only as a sanity check. It helps detect whether a 10k BPE vocabulary is memorising article-specific whole words.",
            "vocab_budget": f"{vocab_size} total tokens including {len(SPECIAL_TOKENS)} special tokens",
            "ratio_definition": "X = total BPE tokens after grapheme-cluster BPE encoding of the full cleaned article / whitespace word units in the same full cleaned article",
            "faithful_unit_definition": "One contiguous Unicode letter/mark/number run, or one visible non-whitespace punctuation/symbol character.",
            "faithful_ratio_definition": "faithful X = total BPE tokens / faithful units in the same full cleaned article",
            "score_formula": "1000 / (max(X1..X4) - min(X1..X4))",
        },
        "summary": {
            "requested_vocab_size": vocab_size,
            "actual_vocab_size": len(model.vocab),
            "special_token_count": len(SPECIAL_TOKENS),
            "learned_token_count": max(0, len(model.vocab) - len(SPECIAL_TOKENS)),
            "merge_count": len(model.merges),
            "english_constraint_x_le_1_2": next((float(r["fertility_x"]) <= 1.2 for r in per_language if r["code"] == "en"), False),
            "english_x": next((float(r["fertility_x"]) for r in per_language if r["code"] == "en"), None),
            "selected_quotas": optimizer_diagnostics.get("selected_quotas", {}),
            "optimization_strategy": optimizer_diagnostics.get("strategy", "constrained_greedy_bpe"),
            "selected_config": optimizer_diagnostics.get("config_name", ""),
            "max_x": max_x,
            "min_x": min_x,
            "delta_x": delta,
            "score": score,
            "score_display": score_display,
            "sorted_by_x_desc": [x["language"] for x in sorted_langs],
        },
        "faithful_unit_summary": faithful_summary,
        "heldout_diagnostic_summary": {
            "max_x": held_max,
            "min_x": held_min,
            "delta_x": held_delta,
            "score": held_score,
            "score_display": held_score_display,
            "sorted_by_x_desc": [x["language"] for x in held_sorted],
        },
        "optimizer_diagnostics": optimizer_diagnostics,
        "quota_search_diagnostics": optimizer_diagnostics,
        "per_language": per_language,
        "full_page_per_language": per_language,
        "heldout_diagnostic_per_language": heldout_per_language,
    }

    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for lang in LANGUAGES:
        code = str(lang["code"])
        (corpus_dir / f"{code}.txt").write_text(corpus[code], encoding="utf-8")
        (split_dir / f"{code}.train.txt").write_text(train_corpus[code], encoding="utf-8")
        (split_dir / f"{code}.eval.txt").write_text(eval_corpus[code], encoding="utf-8")

    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "validation": metrics_json["validation"],
                "whitespace_word_summary": metrics_json["summary"],
                "faithful_unit_summary": faithful_summary,
                "per_language": per_language,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "vocab.txt").open("w", encoding="utf-8") as f:
        f.write("id\ttoken\tvisible\tfrequency_full\tis_special\tlength_chars\n")
        for rec in vocab_records:
            f.write(f"{rec['id']}\t{rec['token']}\t{rec['visible']}\t{rec['frequency_full']}\t{rec['is_special']}\t{rec['length_chars']}\n")
    with (output_dir / "merges.txt").open("w", encoding="utf-8") as f:
        f.write("rank\tleft\tright\tmerged\n")
        for i, (a, b, merged) in enumerate(model.merges):
            f.write(f"{i}\t{a}\t{b}\t{merged}\n")
    return metrics_json

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="public/data", help="Output directory for metrics/tokenizer files")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--force-fallback", action="store_true", help="Debug only: use embedded sample text. Do not use for submission.")
    parser.add_argument("--strict-live", action="store_true", default=True, help="Default: fail if live Wikipedia fetch fails; kept for compatibility.")
    parser.add_argument("--train-fraction", type=float, default=0.80, help="Fraction of each article used to train raw baseline BPE before held-out evaluation")
    args = parser.parse_args(argv)

    metrics = build_data(Path(args.out), vocab_size=args.vocab_size, force_fallback=args.force_fallback, strict_live=args.strict_live, train_fraction=args.train_fraction)
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

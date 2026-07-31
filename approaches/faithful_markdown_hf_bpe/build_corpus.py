#!/usr/bin/env python3
"""Fetch Wikipedia REST HTML and save reproducible faithful-Markdown snapshots."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from common import LANGUAGES, sha256_text


def normalize_markdown(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def fetch_language(code: str, timeout: int = 60) -> tuple[str, dict[str, object]]:
    language = LANGUAGES[code]
    host = str(language["host"])
    title = str(language["title"])
    endpoint = f"https://{host}/api/rest_v1/page/html/{quote(title, safe='')}"
    response = requests.get(endpoint, timeout=timeout, headers={
        "User-Agent": "IndiaBPEFaithfulMarkdown/1.0 (reproducible student project)",
        "Accept": "text/html; charset=utf-8",
    })
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    for element in soup.select("script, style, meta, noscript"):
        element.decompose()
    base = f"https://{host}/"
    for element in soup.find_all(href=True):
        element["href"] = urljoin(base, str(element["href"]))
    for element in soup.find_all(src=True):
        element["src"] = urljoin(base, str(element["src"]))
    body = soup.body or soup
    faithful = normalize_markdown(markdownify(str(body), heading_style="ATX", bullets="-"))
    metadata = {
        "code": code,
        "language": language["name"],
        "wiki_title": title,
        "source_url": f"https://{host}/wiki/{quote(title)}",
        "rest_html_url": endpoint,
        "fetched_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "response_etag": response.headers.get("ETag"),
        "response_last_modified": response.headers.get("Last-Modified"),
        "characters": len(faithful),
        "sha256": sha256_text(faithful),
    }
    return faithful, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", required=True, help="comma-separated codes, e.g. en,hi,te,kn")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    codes = [code.strip() for code in args.languages.split(",") if code.strip()]
    unknown = set(codes) - set(LANGUAGES)
    if unknown:
        raise ValueError(f"unknown language codes: {sorted(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"method": "Wikipedia REST HTML to faithful Markdown", "languages": []}
    for code in codes:
        text, metadata = fetch_language(code)
        (args.out / f"{code}.faithful.md").write_text(text, encoding="utf-8")
        (args.out / f"{code}.faithful.txt").write_text(text, encoding="utf-8")
        (args.out / f"{code}.meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["languages"].append(metadata)
        print(f"{code}: {metadata['characters']} characters, {metadata['sha256']}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

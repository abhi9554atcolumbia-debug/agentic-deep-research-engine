"""
fetch_parse.py
Layer 5 (Web + PDF Fetcher) + Layer 6 (Parser + Chunker).

Given a source URL:
  - Detects whether it's a PDF or HTML page.
  - Fetches and extracts clean text.
  - Splits the text into overlapping chunks with metadata.
"""

import io
import requests
import trafilatura
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Engine Bot; for academic hackathon project)"
}
TIMEOUT = 15  # seconds


def fetch_and_extract_text(url: str) -> str:
    """
    Fetch a URL and return extracted plain text.
    Handles both HTML pages and PDFs. Returns "" on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [fetch error] {url}: {e}")
        return ""

    content_type = resp.headers.get("Content-Type", "").lower()

    # --- PDF handling ---
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(resp.content))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"  [pdf parse error] {url}: {e}")
            return ""

    # --- HTML handling ---
    try:
        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
        return extracted or ""
    except Exception as e:
        print(f"  [html parse error] {url}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping chunks (character-based, simple and fast).
    Returns a list of chunk strings (whitespace-trimmed, empty ones dropped).
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def fetch_and_chunk(source: dict):
    """
    Given a source dict {"url", "title", "snippet"}, fetch and chunk its content.
    Returns a list of chunk dicts:
      [{"text": str, "url": str, "title": str, "chunk_index": int}, ...]
    Returns [] if fetching/parsing fails (caller can fall back to the snippet).
    """
    url = source["url"]
    text = fetch_and_extract_text(url)

    if not text:
        # Fallback: use the search snippet as a single "chunk" so the
        # source isn't lost entirely, just marked as low-detail.
        if source.get("snippet"):
            return [{
                "text": source["snippet"],
                "url": url,
                "title": source.get("title", ""),
                "chunk_index": 0,
                "fallback_snippet_only": True,
            }]
        return []

    raw_chunks = chunk_text(text)
    return [
        {
            "text": c,
            "url": url,
            "title": source.get("title", ""),
            "chunk_index": i,
            "fallback_snippet_only": False,
        }
        for i, c in enumerate(raw_chunks)
    ]


if __name__ == "__main__":
    # Quick manual test
    test_source = {"url": "https://en.wikipedia.org/wiki/Indoor_air_quality", "title": "test", "snippet": ""}
    chunks = fetch_and_chunk(test_source)
    print(f"Got {len(chunks)} chunks")
    if chunks:
        print(chunks[0]["text"][:300])

"""
evidence_extractor.py
Layer 9 (Evidence Extractor).

For the top-N VGRH-ranked chunks of a sub-question, ask the LLM to
pull out concrete claims/facts as structured evidence items:

  {
    "claim": "...",
    "evidence_snippet": "...",   # short quote/paraphrase grounding the claim
    "source_url": "...",
    "source_title": "...",
    "vgrh_score": 0.0-1.0,
    "sub_question": "..."
  }

These evidence items are the atomic units passed to the claim
verifier and report generator.
"""

from llm import call_llm_json
from config import TOP_N_FOR_EVIDENCE

EXTRACTION_PROMPT = """You are extracting evidence for a research report.

Sub-question being answered: "{sub_question}"

Below are text chunks from a source. Extract up to 3 concrete, specific
CLAIMS from this text that help answer the sub-question. Only extract
claims that are actually supported by the text below -- do not invent
information. Skip vague or boilerplate text. If the chunk contains no
useful claims, return an empty array.

For each claim, include a short "evidence_snippet": a short paraphrase
(max ~25 words) of the specific part of the text that supports the claim.
Do NOT copy long verbatim passages.

Source title: {title}
Source URL: {url}

Text:
{text}

Respond with ONLY a JSON array (can be empty []):
[
  {{"claim": "...", "evidence_snippet": "..."}},
  ...
]
"""


def extract_evidence_for_chunk(sub_question: str, chunk: dict):
    """
    Returns a list of evidence dicts extracted from a single chunk.
    """
    prompt = EXTRACTION_PROMPT.format(
        sub_question=sub_question,
        title=chunk.get("title", ""),
        url=chunk["url"],
        text=chunk["text"][:1500],
    )

    try:
        items = call_llm_json(prompt, max_tokens=600)
        if not isinstance(items, list):
            return []
    except Exception as e:
        print(f"  [evidence extraction error] {chunk['url']}: {e}")
        return []

    evidence = []
    for item in items:
        claim = item.get("claim", "").strip()
        snippet = item.get("evidence_snippet", "").strip()
        if not claim:
            continue
        evidence.append({
            "claim": claim,
            "evidence_snippet": snippet,
            "source_url": chunk["url"],
            "source_title": chunk.get("title", ""),
            "vgrh_score": chunk.get("vgrh_score", 0.0),
            "vgrh_breakdown": chunk.get("vgrh_breakdown", {}),
            "sub_question": sub_question,
        })
    return evidence


def extract_evidence_for_subquestion(sub_question: str, ranked_chunks: list, top_n: int = TOP_N_FOR_EVIDENCE):
    """
    Runs evidence extraction over the top_n VGRH-ranked chunks for a
    sub-question. Returns a flat list of evidence dicts.
    """
    all_evidence = []
    for chunk in ranked_chunks[:top_n]:
        all_evidence.extend(extract_evidence_for_chunk(sub_question, chunk))
    return all_evidence


if __name__ == "__main__":
    # Quick manual test
    sample_chunk = {
        "text": "EIA data shows US crude production reached 13.2 million barrels per day in 2025, "
                "a 4% increase year-on-year, driven mainly by shale output in the Permian Basin. "
                "Meanwhile OPEC+ extended voluntary production cuts of 2.2 million bpd through Q2 2026.",
        "url": "https://www.eia.gov/example",
        "title": "EIA Example",
        "vgrh_score": 0.8,
        "vgrh_breakdown": {},
    }
    ev = extract_evidence_for_chunk("What supply factors affect oil production?", sample_chunk)
    for e in ev:
        print(e)

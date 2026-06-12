"""
score_and_extract.py
Combines Layer 8 (VGRH Ranker) + Layer 9 (Evidence Extractor) into a
SINGLE LLM call per sub-question, instead of 1 + up to N calls.

This is critical when running on rate-limited free-tier LLM APIs
(e.g. Gemini free tier: 20 requests/day) -- the original design used
1 (VGRH batch) + up to TOP_N_FOR_EVIDENCE (evidence) calls PER
sub-question, which alone could exceed a 20/day quota.

For each sub-question, ALL retrieved chunks are sent in ONE prompt.
The LLM returns, per chunk: veracity_llm + helpfulness scores AND
0-3 extracted evidence claims. Veracity (domain part), Grounding,
and Relevance are still computed locally (no LLM needed) exactly as
before in vgrh_ranker.py.
"""

from llm import call_llm_json
from domain_trust import domain_trust_score
from vgrh_ranker import grounding_score
from config import VGRH_WEIGHTS, VERACITY_DOMAIN_WEIGHT, VERACITY_LLM_WEIGHT, TOP_N_FOR_EVIDENCE

COMBINED_PROMPT = """You are evaluating evidence chunks retrieved for a research sub-question.

Sub-question: "{sub_question}"

For EACH chunk below, do TWO things:

1. Rate on a 1-5 scale:
   - "veracity_llm": based on CONTENT only (not source name) -- specific
     data/figures/dates/attributed sources = high; vague, promotional,
     unsourced, sensational = low.
   - "helpfulness": does this chunk give specific, useful info that
     helps answer the sub-question? High = concrete facts/mechanisms
     directly relevant. Low = generic/tangential/boilerplate.

2. Extract up to 3 concrete CLAIMS from the chunk that help answer the
   sub-question, each with a short "evidence_snippet" (paraphrase,
   max ~25 words, do NOT copy verbatim). Prioritize direct answers, but
   also extract useful background context, stated motives, official
   justifications, reactions, legal claims, risks, and consequences when
   those help answer the sub-question. Only return an empty array if the
   chunk truly gives no usable information for this sub-question.

Chunks:
{chunks_block}

Respond with ONLY a JSON array, one object per chunk, SAME ORDER:
[
  {{
    "veracity_llm": <1-5>,
    "helpfulness": <1-5>,
    "evidence": [
      {{"claim": "...", "evidence_snippet": "..."}}
    ]
  }},
  ...
]
"""


def score_and_extract(sub_question: str, chunks: list, top_n_for_evidence: int = TOP_N_FOR_EVIDENCE):
    """
    Returns (ranked_chunks, evidence_items):
      - ranked_chunks: input chunks augmented with vgrh_breakdown +
        vgrh_score, sorted descending by vgrh_score.
      - evidence_items: flat list of evidence dicts (claim,
        evidence_snippet, source_url, source_title, vgrh_score,
        vgrh_breakdown, sub_question), extracted only from the
        top_n_for_evidence ranked chunks.
    """
    if not chunks:
        return [], []

    chunks_block = "\n\n".join(
        f"Chunk {i+1}:\nTitle: {c.get('title', '')}\nURL: {c.get('url', '')}\nText: {c['text'][:900]}"
        for i, c in enumerate(chunks)
    )
    prompt = COMBINED_PROMPT.format(sub_question=sub_question, chunks_block=chunks_block)

    try:
        results = call_llm_json(prompt, max_tokens=2048)
        if not isinstance(results, list) or len(results) != len(chunks):
            raise ValueError("combined score/extract length mismatch")
    except Exception as e:
        print(f"  [score_and_extract fallback] {e}")
        results = [{"veracity_llm": 3, "helpfulness": 3, "evidence": []} for _ in chunks]

    ranked = []
    for chunk, r in zip(chunks, results):
        ver_llm = max(0.0, min(1.0, float(r.get("veracity_llm", 3)) / 5.0))
        helpfulness = max(0.0, min(1.0, float(r.get("helpfulness", 3)) / 5.0))

        domain_ver = domain_trust_score(chunk["url"])
        veracity = (VERACITY_DOMAIN_WEIGHT * domain_ver) + (VERACITY_LLM_WEIGHT * ver_llm)
        grounding = grounding_score(sub_question, chunk["text"])
        relevance = max(0.0, chunk.get("retrieval_score", 0.0))

        breakdown = {
            "veracity": round(veracity, 3),
            "grounding": round(grounding, 3),
            "relevance": round(relevance, 3),
            "helpfulness": round(helpfulness, 3),
        }
        total = (
            VGRH_WEIGHTS["veracity"] * veracity
            + VGRH_WEIGHTS["grounding"] * grounding
            + VGRH_WEIGHTS["relevance"] * relevance
            + VGRH_WEIGHTS["helpfulness"] * helpfulness
        )

        new_chunk = dict(chunk)
        new_chunk["vgrh_breakdown"] = breakdown
        new_chunk["vgrh_score"] = round(total, 3)
        new_chunk["_raw_evidence"] = r.get("evidence", []) or []
        ranked.append(new_chunk)

    ranked.sort(key=lambda c: c["vgrh_score"], reverse=True)

    # Extract evidence only from the top-N ranked chunks
    evidence_items = []
    for chunk in ranked[:top_n_for_evidence]:
        for item in chunk.get("_raw_evidence", []):
            claim = (item.get("claim") or "").strip()
            snippet = (item.get("evidence_snippet") or "").strip()
            if not claim:
                continue
            evidence_items.append({
                "claim": claim,
                "evidence_snippet": snippet,
                "source_url": chunk["url"],
                "source_title": chunk.get("title", ""),
                "vgrh_score": chunk["vgrh_score"],
                "vgrh_breakdown": chunk["vgrh_breakdown"],
                "sub_question": sub_question,
            })

    # Clean up internal field
    for c in ranked:
        c.pop("_raw_evidence", None)

    return ranked, evidence_items


if __name__ == "__main__":
    sample_chunks = [
        {
            "text": "EIA data shows US crude production reached 13.2 million barrels per day in 2025, up 4% year-on-year, driven by Permian Basin shale output.",
            "url": "https://www.eia.gov/example",
            "title": "EIA Example",
            "retrieval_score": 0.65,
        },
        {
            "text": "Oil prices might go up or down depending on many factors, hard to say.",
            "url": "https://randomblog.example.com/oil",
            "title": "Random Blog",
            "retrieval_score": 0.40,
        },
    ]
    ranked, evidence = score_and_extract("What factors affect oil production?", sample_chunks)
    for c in ranked:
        print(c["vgrh_score"], c["vgrh_breakdown"], "-", c["url"])
    print("\nEvidence:")
    for e in evidence:
        print(e)

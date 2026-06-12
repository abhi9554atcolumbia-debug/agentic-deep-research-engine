"""
vgrh_ranker.py
Layer 8 (VGRH Ranker).

Scores each retrieved chunk on four axes and combines them into a
single weighted "vgrh_score". The breakdown is kept on each chunk so
it can be displayed in the source table (judges explicitly want
explainable ranking, not a black-box number).

  V - Veracity:    is the source/content likely trustworthy?
                   = blend of (a) domain-trust lookup table and
                              (b) LLM judgment of the chunk's content
                                  (specificity, hedging, presence of
                                  data/citations vs. vague claims)
  G - Grounding:   is the chunk's text actually about the specific
                   sub-question, in concrete/lexical terms?
                   = keyword overlap between sub-question and chunk
  R - Relevance:   semantic similarity between sub-question and chunk
                   = retrieval (cosine) score, already computed
  H - Helpfulness: does the chunk add specific, actionable info?
                   = LLM 1-5 rating, normalized to 0-1

To keep LLM calls cheap, veracity_llm + helpfulness for ALL chunks of
one sub-question are scored in a single batched call.
"""

import re
from llm import call_llm_json
from domain_trust import domain_trust_score
from config import VGRH_WEIGHTS, VERACITY_DOMAIN_WEIGHT, VERACITY_LLM_WEIGHT

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "how",
    "do", "does", "did", "in", "on", "of", "to", "for", "and", "or", "but",
    "this", "that", "these", "those", "with", "by", "from", "as", "at",
    "be", "been", "their", "its", "it", "role", "extent", "current",
    "currently", "such", "into", "about", "shifts", "major",
}


def _keywords(text: str):
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def grounding_score(sub_question: str, chunk_text: str) -> float:
    """
    Lexical overlap between sub-question keywords and chunk text.
    Returns fraction (0-1) of sub-question keywords found in the chunk.
    """
    q_words = _keywords(sub_question)
    if not q_words:
        return 0.0
    chunk_words = _keywords(chunk_text)
    overlap = q_words & chunk_words
    return len(overlap) / len(q_words)


BATCH_PROMPT = """You are evaluating evidence chunks retrieved for a research sub-question.

Sub-question: "{sub_question}"

For EACH chunk below, rate two things on a 1-5 scale:

- "veracity_llm": Based on the chunk's CONTENT (not the source name), how
  trustworthy does it seem? High = specific data, figures, dates,
  attributed sources, measured/cautious claims. Low = vague,
  promotional, unsourced, sensational, or speculative language.

- "helpfulness": Does this chunk add specific, useful, actionable
  information that helps answer the sub-question? High = concrete
  facts/figures/mechanisms directly relevant. Low = generic,
  tangential, or boilerplate text.

Chunks:
{chunks_block}

Respond with ONLY a JSON array, one object per chunk, in the same order:
[
  {{"veracity_llm": <1-5>, "helpfulness": <1-5>}},
  ...
]
"""


def _score_batch_with_llm(sub_question: str, chunks: list):
    """
    Returns a list of (veracity_llm_norm, helpfulness_norm) tuples,
    each in [0,1], one per chunk, same order as input.
    Falls back to neutral 0.5 scores for all chunks if the LLM call fails.
    """
    chunks_block = "\n\n".join(
        f"Chunk {i+1}:\n{c['text'][:600]}"
        for i, c in enumerate(chunks)
    )
    prompt = BATCH_PROMPT.format(sub_question=sub_question, chunks_block=chunks_block)

    try:
        results = call_llm_json(prompt, max_tokens=1024)
        if not isinstance(results, list) or len(results) != len(chunks):
            raise ValueError("LLM batch score length mismatch")
        scored = []
        for r in results:
            v = float(r.get("veracity_llm", 3)) / 5.0
            h = float(r.get("helpfulness", 3)) / 5.0
            scored.append((max(0.0, min(1.0, v)), max(0.0, min(1.0, h))))
        return scored
    except Exception as e:
        print(f"  [vgrh llm scoring fallback] {e}")
        return [(0.5, 0.5) for _ in chunks]


def rank_chunks(sub_question: str, chunks: list):
    """
    Given a sub-question and its retrieved chunks (each with
    'retrieval_score' and 'url'/'text'), return the same chunks
    augmented with:
      - vgrh_breakdown: {veracity, grounding, relevance, helpfulness}
      - vgrh_score: weighted total (0-1)
    sorted by vgrh_score descending.
    """
    if not chunks:
        return []

    llm_scores = _score_batch_with_llm(sub_question, chunks)

    ranked = []
    for chunk, (ver_llm, helpfulness) in zip(chunks, llm_scores):
        domain_ver = domain_trust_score(chunk["url"])
        veracity = (VERACITY_DOMAIN_WEIGHT * domain_ver) + (VERACITY_LLM_WEIGHT * ver_llm)

        grounding = grounding_score(sub_question, chunk["text"])
        relevance = max(0.0, chunk.get("retrieval_score", 0.0))  # cosine can be slightly negative

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
        ranked.append(new_chunk)

    ranked.sort(key=lambda c: c["vgrh_score"], reverse=True)
    return ranked


if __name__ == "__main__":
    # Quick manual test
    sample_chunks = [
        {
            "text": "According to EIA data, US crude production reached 13.2 million barrels per day in 2025, up 4% year-on-year.",
            "url": "https://www.eia.gov/outlooks/steo/report/global_oil.php",
            "title": "EIA STEO",
            "retrieval_score": 0.65,
        },
        {
            "text": "Oil prices might go up or down depending on many things, it's complicated and hard to predict.",
            "url": "https://randomblog.example.com/oil",
            "title": "Random Blog",
            "retrieval_score": 0.40,
        },
    ]
    ranked = rank_chunks("What factors affect oil production levels?", sample_chunks)
    for c in ranked:
        print(c["vgrh_score"], c["vgrh_breakdown"], "-", c["url"])

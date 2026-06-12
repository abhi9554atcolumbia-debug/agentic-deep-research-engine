"""
iteration.py
Implements the "Iterate / Refine Queries" feedback loop shown in the
architecture diagram (the dashed loop back from
"Issues / Low Confidence Detected?" to the Research Planner / Query
Generator).

For a sub-question whose findings are missing or low-confidence,
this module:
  1. Asks the LLM to generate 1-2 NEW, more targeted search queries
     (given what's already been tried and what's missing).
  2. Re-runs search -> fetch/chunk -> retrieve -> score_and_extract
     for those new queries only.
  3. Merges the new evidence with the old evidence and re-runs the
     claim verifier.

Capped at ONE iteration per sub-question to keep LLM call counts
bounded (important for free-tier rate limits).
"""

from llm import call_llm_json
from search import search_sources
from fetch_parse import fetch_and_chunk
from retriever import ChunkIndex
from score_and_extract import score_and_extract
from claim_verifier import verify_subquestion_evidence


REFINE_PROMPT = """You are refining a research search for this sub-question:
"{sub_question}"

Searches already tried: {tried_queries}

Current findings so far (may be empty or low-confidence):
{findings_summary}

The findings above are insufficient (missing, low-confidence, or
contradictory). Suggest 1-2 NEW, more specific or differently-angled
search queries that might surface better sources. Avoid repeating the
queries already tried.

Respond with ONLY a JSON array of strings:
["new query 1", "new query 2"]
"""


def needs_refinement(findings: list) -> bool:
    """A sub-question needs refinement if it has no findings, or
    every finding is low-confidence."""
    if not findings:
        return True
    return all(f["confidence"] == "low" for f in findings)


def _summarize_findings(findings: list) -> str:
    if not findings:
        return "(none)"
    lines = []
    for f in findings:
        note = f" - note: {f['note']}" if f.get("note") else ""
        lines.append(f"- [{f['confidence']}] {f['finding']}{note}")
    return "\n".join(lines)


def generate_refinement_queries(sub_question: str, tried_queries: list, findings: list):
    prompt = REFINE_PROMPT.format(
        sub_question=sub_question,
        tried_queries=", ".join(f'"{q}"' for q in tried_queries),
        findings_summary=_summarize_findings(findings),
    )
    try:
        queries = call_llm_json(prompt, max_tokens=300)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str) and q.strip()][:2]
    except Exception as e:
        print(f"  [refinement query generation error] {e}")
    return []


def refine_subquestion(sub_question: str, tried_queries: list, existing_evidence: list,
                        existing_findings: list, seen_urls: set):
    """
    Runs one refinement iteration for a sub-question.

    Returns:
      (new_ranked_chunks, merged_evidence, new_findings, new_queries)

    new_ranked_chunks: VGRH-ranked chunks discovered in this iteration
                       (caller should add these to the global source list)
    merged_evidence:   existing_evidence + newly extracted evidence
    new_findings:      re-verified findings using merged_evidence
    new_queries:       the refinement queries used (for logging/display)
    """
    new_queries = generate_refinement_queries(sub_question, tried_queries, existing_findings)
    if not new_queries:
        return [], existing_evidence, existing_findings, []

    # --- Search + fetch + chunk for the new queries only ---
    new_chunks = []
    for q in new_queries:
        for r in search_sources(q):
            url = r.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            chunks = fetch_and_chunk(r)
            for c in chunks:
                c["sub_question"] = sub_question
                new_chunks.append(c)

    if not new_chunks:
        return [], existing_evidence, existing_findings, new_queries

    # --- Retrieve + score/extract on the new chunks ---
    idx = ChunkIndex(new_chunks)
    idx.build()
    retrieved = idx.retrieve(sub_question)
    ranked, new_evidence = score_and_extract(sub_question, retrieved)

    merged_evidence = existing_evidence + new_evidence
    new_findings = verify_subquestion_evidence(sub_question, merged_evidence)

    return ranked, merged_evidence, new_findings, new_queries

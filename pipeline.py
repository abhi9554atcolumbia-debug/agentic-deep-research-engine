"""
pipeline.py
Core pipeline logic refactored as a GENERATOR that yields progress
events, so the Streamlit UI (app.py) can display live progress
without blocking.

Each yielded item is a dict: {"type": ..., ...}. Possible types:
  - "plan"            : {"plan": [...]}
  - "search"          : {"sub_question": ..., "query": ..., "count": int}
  - "fetch"           : {"url": ..., "n_chunks": int}
  - "ranked"          : {"sub_question": ..., "chunks": [...]}
  - "evidence"        : {"sub_question": ..., "evidence": [...]}
  - "findings"        : {"sub_question": ..., "findings": [...]}
  - "refinement"      : {"sub_question": ..., "queries": [...]}
  - "report"          : {"report_md": ...}
  - "done"            : {"result": {...}}   # final aggregated result
  - "error"           : {"message": ...}
"""

import time
from planner import plan_research
from search import search_sources
from fetch_parse import fetch_and_chunk
from retriever import ChunkIndex
from score_and_extract import score_and_extract
from claim_verifier import verify_subquestion_evidence
from report_generator import generate_report
from iteration import needs_refinement, refine_subquestion


def run_pipeline_stream(user_query: str, enable_iteration: bool = True, call_delay: float = 2.0):
    """
    Generator that runs the full pipeline, yielding progress events.
    The final event has type "done" with the aggregated result dict:
      {
        "query": str,
        "plan": [...],
        "findings_by_subq": {sub_question: [findings...]},
        "all_ranked_chunks": [...],
        "report_md": str,
      }
    """
    try:
        # --- Layer 2 & 3: Plan ---
        plan = plan_research(user_query)
        yield {"type": "plan", "plan": plan}
        time.sleep(call_delay)

        all_ranked_chunks = []
        all_findings_by_subq = {}

        for item in plan:
            sub_q = item["sub_question"]
            tried_queries = list(item["search_queries"])

            # --- Layer 4: Search / Source Discovery ---
            seen_urls = set()
            sub_chunks = []
            for q in tried_queries:
                results = search_sources(q)
                yield {"type": "search", "sub_question": sub_q, "query": q, "count": len(results)}
                for r in results:
                    url = r.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    # --- Layer 5 & 6: Fetch + Parse + Chunk ---
                    chunks = fetch_and_chunk(r)
                    for c in chunks:
                        c["sub_question"] = sub_q
                        sub_chunks.append(c)
                    yield {"type": "fetch", "url": url, "n_chunks": len(chunks)}

            if not sub_chunks:
                all_findings_by_subq[sub_q] = []
                yield {"type": "findings", "sub_question": sub_q, "findings": []}
                continue

            # --- Layer 7: Retriever ---
            idx = ChunkIndex(sub_chunks)
            idx.build()
            retrieved = idx.retrieve(sub_q)

            # --- Layer 8 + 9: VGRH Ranker + Evidence Extractor (1 LLM call) ---
            ranked, evidence = score_and_extract(sub_q, retrieved)
            yield {"type": "ranked", "sub_question": sub_q, "chunks": ranked}
            yield {"type": "evidence", "sub_question": sub_q, "evidence": evidence}
            time.sleep(call_delay)

            # --- Layer 10: Claim Verifier ---
            findings = verify_subquestion_evidence(sub_q, evidence)
            time.sleep(call_delay)

            # --- Iteration / Refine loop (capped at 1 iteration) ---
            if enable_iteration and needs_refinement(findings):
                new_ranked, merged_evidence, new_findings, new_queries = refine_subquestion(
                    sub_q, tried_queries, evidence, findings, seen_urls
                )
                if new_queries:
                    yield {"type": "refinement", "sub_question": sub_q, "queries": new_queries}
                    ranked = ranked + new_ranked
                    findings = new_findings
                    time.sleep(call_delay)

            all_ranked_chunks.extend(ranked)
            all_findings_by_subq[sub_q] = findings
            yield {"type": "findings", "sub_question": sub_q, "findings": findings}

        # --- Layer 11: Report Generator ---
        report_md = generate_report(user_query, plan, all_findings_by_subq, all_ranked_chunks)
        yield {"type": "report", "report_md": report_md}

        yield {"type": "done", "result": {
            "query": user_query,
            "plan": plan,
            "findings_by_subq": all_findings_by_subq,
            "all_ranked_chunks": all_ranked_chunks,
            "report_md": report_md,
        }}

    except Exception as e:
        yield {"type": "error", "message": str(e)}

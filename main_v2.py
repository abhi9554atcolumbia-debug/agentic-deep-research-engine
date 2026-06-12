"""
main.py
Full pipeline orchestrator (Phase 1 + Phase 2).

Pipeline:
  1. User Query
  2. Research Planner (sub-questions)
  3. Query Generator (search queries per sub-question)
  4. Search / Source Discovery (Tavily)
  5. Web + PDF Fetcher
  6. Parser + Chunker
  7. Retriever (embeddings, top-k per sub-question)
  8. VGRH Ranker (Veracity, Grounding, Relevance, Helpfulness)
  9. Evidence Extractor (structured claims from top-ranked chunks)
  10. Claim Verifier (group, cross-check, confidence/contradictions)
  11. Report Generator (final cited Markdown report)

Output: prints the pipeline trace AND writes a Markdown report to
./output_report.md
"""

import sys
import time
from planner import plan_research
from search import search_sources
from fetch_parse import fetch_and_chunk
from retriever import ChunkIndex
from score_and_extract import score_and_extract
from claim_verifier import verify_subquestion_evidence
from report_generator import generate_report


def run_pipeline(user_query: str, output_path: str = "output_report.md"):
    print(f"\n=== USER QUERY ===\n{user_query}\n")

    # --- Layer 2 & 3: Plan + query generation ---
    print("=== PLANNING (sub-questions + search queries) ===")
    plan = plan_research(user_query)
    for item in plan:
        print(f"\n- Sub-question: {item['sub_question']}")
        for q in item["search_queries"]:
            print(f"    search query: {q}")

    # --- Layer 4: Search / source discovery ---
    print("\n=== SOURCE DISCOVERY ===")
    all_sources = []  # list of (sub_question, source_dict)
    seen_urls = set()
    for item in plan:
        sub_q = item["sub_question"]
        for sq in item["search_queries"]:
            results = search_sources(sq)
            print(f"  '{sq}' -> {len(results)} results")
            for r in results:
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_sources.append((sub_q, r))

    print(f"\nTotal unique sources discovered: {len(all_sources)}")

    # --- Layer 5 & 6: Fetch + parse + chunk ---
    print("\n=== FETCHING & CHUNKING ===")
    all_chunks_flat = []
    for sub_q, source in all_sources:
        chunks = fetch_and_chunk(source)
        for c in chunks:
            c["sub_question"] = sub_q
            all_chunks_flat.append(c)
        print(f"  {source['url']} -> {len(chunks)} chunks")

    print(f"\nTotal chunks collected: {len(all_chunks_flat)}")

    if not all_chunks_flat:
        print("\nNo chunks were collected. Check your TAVILY_API_KEY, "
              "network access, and that sources are reachable.")
        return

    # --- Layer 7, 8, 9, 10 per sub-question ---
    all_findings_by_subq = {}
    all_ranked_chunks = []  # for the source table

    for item in plan:
        sub_q = item["sub_question"]
        sub_chunks = [c for c in all_chunks_flat if c["sub_question"] == sub_q]

        print(f"\n--- {sub_q} ---")

        if not sub_chunks:
            print("  No chunks for this sub-question, skipping.")
            all_findings_by_subq[sub_q] = []
            continue

        # Layer 7: Retriever
        sub_index = ChunkIndex(sub_chunks)
        sub_index.build()
        retrieved = sub_index.retrieve(sub_q)
        print(f"  Retrieved {len(retrieved)} chunks")

        # Layer 8 + 9 combined: VGRH Ranker + Evidence Extractor (1 LLM call)
        ranked, evidence = score_and_extract(sub_q, retrieved)
        all_ranked_chunks.extend(ranked)
        for c in ranked:
            print(f"    [vgrh={c['vgrh_score']:.3f}] {c['vgrh_breakdown']} - {c['url']}")
        print(f"  Extracted {len(evidence)} evidence items")

        time.sleep(3)  # space out LLM calls to help with rate limits

        # Layer 10: Claim Verifier
        findings = verify_subquestion_evidence(sub_q, evidence)
        all_findings_by_subq[sub_q] = findings
        for f in findings:
            print(f"    [{f['confidence']}] {f['finding']} (sources: {f['num_sources']})")
            if f["note"]:
                print(f"        note: {f['note']}")

        time.sleep(3)  # space out LLM calls to help with rate limits

    # --- Layer 11: Report Generator ---
    print("\n=== GENERATING REPORT ===")
    report_md = generate_report(user_query, plan, all_findings_by_subq, all_ranked_chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nReport written to: {output_path}")

    return {
        "query": user_query,
        "plan": plan,
        "findings_by_subq": all_findings_by_subq,
        "report_path": output_path,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Compare three low-cost methods for improving air quality in classrooms."

    run_pipeline(query)

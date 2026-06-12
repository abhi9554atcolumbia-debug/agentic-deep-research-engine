"""
report_generator.py
Layer 11 (Report Generator).

Takes the research plan + all verified findings (with evidence and
confidence) and produces a structured Markdown report matching the
brief's suggested output format:

  - Executive summary
  - Research plan
  - Source table
  - Evidence table
  - Final report (synthesis with citations)
  - Limitations

By default, the executive summary and synthesis are generated locally
from verified findings to avoid spending extra API calls at the end of
a run. Set USE_LLM_REPORT=true in the environment to use one combined
LLM call for those prose sections.
"""

from datetime import datetime
from config import USE_LLM_REPORT

REPORT_SECTIONS_PROMPT = """You are writing two prose sections of a research report.

Original research question: "{query}"

Below are VERIFIED FINDINGS, organized by sub-question. Each finding
has a confidence level (high/medium/low) and a citation marker like [F3]
that you MUST use in the synthesis when referencing it.

{findings_block}

Return ONLY valid JSON in this shape:
{{
  "executive_summary": "3-5 sentences that directly answer the research question, based ONLY on high and medium confidence findings. No citations.",
  "synthesis": "4-8 plain prose paragraphs that answer the question by weaving findings together with citation markers."
}}

Rules:
- Every factual claim in synthesis MUST be immediately followed by its
  citation marker, e.g. "...rose by 4% [F3]."
- Do NOT introduce any fact, number, or claim that is not in the
  findings above.
- When findings have "low" confidence or contradictions, say so
  explicitly (e.g. "sources disagree on...").
"""


def _format_findings_block(findings: list, with_ids: bool = True):
    lines = []
    for i, f in enumerate(findings):
        marker = f"[F{i+1}] " if with_ids else ""
        sources = ", ".join(sorted({e["source_title"] or e["source_url"] for e in f["supporting_evidence"]}))
        line = f"{marker}({f['confidence'].upper()} confidence, {f['num_sources']} source(s): {sources}) {f['finding']}"
        if f.get("note"):
            line += f" [Note: {f['note']}]"
        lines.append(line)
    return "\n".join(lines)


def _flatten_findings(plan: list, all_findings_by_subq: dict):
    flat_findings = []
    for item in plan:
        flat_findings.extend(all_findings_by_subq.get(item["sub_question"], []))
    return flat_findings


def calculate_research_metrics(plan: list, all_findings_by_subq: dict, all_chunks: list):
    """
    Returns dashboard metrics for the report/UI.

    Research confidence blends claim confidence with coverage. This keeps
    a run with many verified claims in only one sub-question from looking
    overconfident for the whole research question.
    """
    flat_findings = _flatten_findings(plan, all_findings_by_subq)
    sources_analyzed = len({c.get("url") for c in all_chunks if c.get("url")})
    evidence_items = sum(len(f.get("supporting_evidence", [])) for f in flat_findings)
    verified_claims = len(flat_findings)

    confidence_weights = {"high": 1.0, "medium": 0.7, "low": 0.35}
    if flat_findings:
        avg_claim_confidence = sum(
            confidence_weights.get(f.get("confidence"), 0.35)
            for f in flat_findings
        ) / len(flat_findings)
    else:
        avg_claim_confidence = 0.0

    covered_subquestions = sum(
        1 for item in plan
        if all_findings_by_subq.get(item["sub_question"], [])
    )
    coverage = covered_subquestions / len(plan) if plan else 0.0
    research_confidence = round((0.7 * avg_claim_confidence + 0.3 * coverage) * 100)

    return {
        "sources_analyzed": sources_analyzed,
        "evidence_items": evidence_items,
        "verified_claims": verified_claims,
        "research_confidence": research_confidence,
    }


def _build_metrics_dashboard(metrics: dict):
    return "\n".join([
        "| Metric | Value |",
        "|---|---:|",
        f"| Sources Analyzed | {metrics['sources_analyzed']} |",
        f"| Evidence Items | {metrics['evidence_items']} |",
        f"| Verified Claims | {metrics['verified_claims']} |",
        f"| Research Confidence | {metrics['research_confidence']}% |",
    ])


def _build_local_exec_summary(query: str, findings: list):
    high_med = [f for f in findings if f["confidence"] in ("high", "medium")]
    usable = high_med or findings

    if not usable:
        return (
            "No verified findings were extracted, so the report cannot make a "
            "confident answer to the research question yet."
        )

    first = usable[0]["finding"].rstrip(".")
    summary = (
        f"For the question \"{query}\", the strongest verified evidence points "
        f"to this finding: {first}."
    )

    if len(usable) > 1:
        other_findings = "; ".join(f["finding"].rstrip(".") for f in usable[1:3])
        summary += f" Additional verified findings show that {other_findings}."

    low_count = sum(1 for f in findings if f["confidence"] == "low")
    if low_count:
        summary += (
            f" {low_count} finding(s) were low-confidence or contradictory, so "
            "those parts should be treated cautiously."
        )

    return summary


def _build_local_synthesis(plan: list, all_findings_by_subq: dict):
    paragraphs = []
    citation_idx = 1

    for item in plan:
        sub_q = item["sub_question"]
        findings = all_findings_by_subq.get(sub_q, [])
        if not findings:
            paragraphs.append(
                f"For the sub-question \"{sub_q}\", the pipeline did not extract "
                "verified findings from the retrieved sources."
            )
            continue

        sentences = []
        for f in findings:
            sentence = f"{f['finding'].rstrip('.')} [F{citation_idx}]."
            if f["confidence"] == "low":
                sentence += " This finding is low-confidence"
                if f.get("note"):
                    sentence += f" because {f['note'].rstrip('.')}"
                sentence += "."
            elif f.get("note"):
                sentence += f" Note: {f['note'].rstrip('.')}."
            sentences.append(sentence)
            citation_idx += 1

        paragraphs.append(f"On \"{sub_q}\", " + " ".join(sentences))

    if not paragraphs:
        return "No verified findings were available to synthesize a report."

    return "\n\n".join(paragraphs)


def _generate_report_sections(query: str, findings_block: str, flat_findings: list,
                              plan: list, all_findings_by_subq: dict):
    if USE_LLM_REPORT and flat_findings:
        try:
            from llm import call_llm_json

            sections = call_llm_json(
                REPORT_SECTIONS_PROMPT.format(query=query, findings_block=findings_block),
                max_tokens=1800,
            )
            exec_summary = (sections.get("executive_summary") or "").strip()
            synthesis = (sections.get("synthesis") or "").strip()
            if exec_summary and synthesis:
                return exec_summary, synthesis
        except Exception as e:
            print(f"  [report prose fallback] {e}")

    return (
        _build_local_exec_summary(query, flat_findings),
        _build_local_synthesis(plan, all_findings_by_subq),
    )


def _build_source_table(all_chunks: list):
    """
    Builds a deduplicated source table from all retrieved/ranked chunks.
    Shows the BEST (max) VGRH score seen for each unique URL.
    """
    by_url = {}
    for c in all_chunks:
        url = c["url"]
        score = c.get("vgrh_score", 0.0)
        if url not in by_url or score > by_url[url]["vgrh_score"]:
            by_url[url] = {
                "title": c.get("title", ""),
                "url": url,
                "vgrh_score": score,
                "vgrh_breakdown": c.get("vgrh_breakdown", {}),
                "fallback": c.get("fallback_snippet_only", False),
            }

    rows = sorted(by_url.values(), key=lambda x: x["vgrh_score"], reverse=True)

    lines = ["| Source | URL | VGRH Score | V / G / R / H | Notes |",
             "|---|---|---|---|---|"]
    for r in rows:
        b = r["vgrh_breakdown"]
        vgrh_str = f"{b.get('veracity','-')} / {b.get('grounding','-')} / {b.get('relevance','-')} / {b.get('helpfulness','-')}" if b else "-"
        notes = "snippet-only (full page not fetchable)" if r["fallback"] else ""
        title = (r["title"] or r["url"])[:60].replace("|", "-")
        lines.append(f"| {title} | {r['url']} | {r['vgrh_score']:.3f} | {vgrh_str} | {notes} |")

    return "\n".join(lines)


def _build_evidence_table(all_findings: list):
    lines = ["| # | Claim | Evidence Snippet | Source | Confidence |",
             "|---|---|---|---|---|"]
    idx = 1
    for f in all_findings:
        for e in f["supporting_evidence"]:
            claim = e["claim"][:100].replace("|", "-")
            snippet = e["evidence_snippet"][:100].replace("|", "-")
            source = (e["source_title"] or e["source_url"])[:40].replace("|", "-")
            lines.append(f"| {idx} | {claim} | {snippet} | {source} | {f['confidence']} |")
            idx += 1
    return "\n".join(lines)


def _build_research_plan_section(plan: list):
    lines = []
    for item in plan:
        lines.append(f"- **{item['sub_question']}**")
        for q in item["search_queries"]:
            lines.append(f"  - search query: `{q}`")
    return "\n".join(lines)


def _build_limitations_section(plan: list, all_findings_by_subq: dict, all_chunks: list):
    lines = []

    # Sub-questions with no findings at all
    for item in plan:
        sub_q = item["sub_question"]
        findings = all_findings_by_subq.get(sub_q, [])
        if not findings:
            lines.append(f"- No verified findings were extracted for: \"{sub_q}\". "
                          f"This may need manual follow-up research.")
            continue
        low_conf = [f for f in findings if f["confidence"] == "low"]
        for f in low_conf:
            note = f" ({f['note']})" if f["note"] else ""
            lines.append(f"- Low-confidence finding for \"{sub_q}\": \"{f['finding']}\"{note}")

    # Sources that couldn't be fully fetched
    fallback_urls = sorted({c["url"] for c in all_chunks if c.get("fallback_snippet_only")})
    if fallback_urls:
        lines.append(f"- {len(fallback_urls)} source(s) could not be fully fetched (paywalled or "
                      f"blocked); only search-result snippets were used for these, which limits "
                      f"the depth of evidence drawn from them.")

    if not lines:
        lines.append("- No major limitations identified; all sub-questions had at least one "
                      "medium/high confidence finding.")

    return "\n".join(lines)


def generate_report(query: str, plan: list, all_findings_by_subq: dict, all_chunks: list) -> str:
    """
    query: original user query (str)
    plan: output of planner.plan_research()
    all_findings_by_subq: {sub_question: [finding dicts, ...]}
    all_chunks: flat list of ALL vgrh-ranked chunks (for the source table)

    Returns the full report as a Markdown string.
    """
    # Flatten findings, preserving sub-question grouping for citation numbering
    flat_findings = _flatten_findings(plan, all_findings_by_subq)

    findings_block = _format_findings_block(flat_findings)
    metrics = calculate_research_metrics(plan, all_findings_by_subq, all_chunks)
    metrics_dashboard = _build_metrics_dashboard(metrics)

    exec_summary, synthesis = _generate_report_sections(
        query, findings_block, flat_findings, plan, all_findings_by_subq
    )

    # --- Build citation key (maps [Fn] -> source list) ---
    citation_lines = []
    for i, f in enumerate(flat_findings):
        sources = ", ".join(sorted({f"[{e['source_title'] or e['source_url']}]({e['source_url']})" for e in f["supporting_evidence"]}))
        citation_lines.append(f"- **[F{i+1}]** ({f['confidence']} confidence) {f['finding']} — Sources: {sources}")
    citation_key = "\n".join(citation_lines) if citation_lines else "No findings."

    source_table = _build_source_table(all_chunks)
    evidence_table = _build_evidence_table(flat_findings)
    research_plan_section = _build_research_plan_section(plan)
    limitations_section = _build_limitations_section(plan, all_findings_by_subq, all_chunks)

    report = f"""# Research Report

**Query:** {query}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## Executive Summary

{exec_summary}

---

## Research Metrics

{metrics_dashboard}

---

## Research Plan

{research_plan_section}

---

## Final Report

{synthesis}

### Citation Key

{citation_key}

---

## Source Table

{source_table}

---

## Evidence Table

{evidence_table}

---

## Limitations

{limitations_section}
"""
    return report

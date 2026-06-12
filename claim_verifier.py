"""
claim_verifier.py
Layer 10 (Claim Verifier).

Takes all extracted evidence items for a sub-question and:
  - Groups them into distinct "findings" (de-duplicating claims that
    say the same thing from different sources).
  - Flags contradictions between sources.
  - Assigns a confidence level (high/medium/low) per finding based on
    (a) how many independent sources support it and
    (b) the average VGRH score of supporting evidence.

Output: a list of verified findings, each with the supporting
evidence items attached -- this is what the report generator
consumes directly.
"""

from llm import call_llm_json
from config import MIN_SOURCES_HIGH_CONFIDENCE

VERIFY_PROMPT = """You are verifying research findings for sub-question:
"{sub_question}"

Below is a numbered list of claims extracted from different sources.
Some claims may be duplicates/restatements of the same finding from
different sources. Some may contradict each other.

Claims:
{claims_block}

Group these into distinct FINDINGS. For each finding:
- Write a clear "finding" statement (your own words, synthesizing
  the claim(s)).
- List the claim numbers that support this finding as "supporting_claim_ids"
  (array of integers).
- If any claims CONTRADICT this finding (state the opposite or a
  meaningfully different number/conclusion), list those claim numbers
  in "contradicting_claim_ids" (array of integers, can be empty).
- Add a one-sentence "note" explaining any contradiction or uncertainty,
  or "" if none.

Respond with ONLY a JSON array:
[
  {{
    "finding": "...",
    "supporting_claim_ids": [1, 3],
    "contradicting_claim_ids": [],
    "note": ""
  }},
  ...
]
"""


def _confidence_label(num_sources: int, avg_vgrh: float, has_contradiction: bool) -> str:
    if has_contradiction:
        return "low"
    if num_sources >= MIN_SOURCES_HIGH_CONFIDENCE and avg_vgrh >= 0.55:
        return "high"
    if num_sources >= 1 and avg_vgrh >= 0.4:
        return "medium"
    return "low"


def verify_subquestion_evidence(sub_question: str, evidence_items: list):
    """
    evidence_items: flat list of evidence dicts from evidence_extractor
                    (all for this sub-question).

    Returns a list of verified finding dicts:
      {
        "finding": str,
        "confidence": "high" | "medium" | "low",
        "num_sources": int,
        "supporting_evidence": [evidence dicts...],
        "contradicting_evidence": [evidence dicts...],
        "note": str,
        "sub_question": str,
      }
    """
    if not evidence_items:
        return []

    claims_block = "\n".join(
        f"{i+1}. {item['claim']} (source: {item['source_title'] or item['source_url']})"
        for i, item in enumerate(evidence_items)
    )
    prompt = VERIFY_PROMPT.format(sub_question=sub_question, claims_block=claims_block)

    try:
        groups = call_llm_json(prompt, max_tokens=1024)
        if not isinstance(groups, list):
            raise ValueError("verifier did not return a list")
    except Exception as e:
        print(f"  [claim verification fallback] {e}")
        # Fallback: treat every claim as its own finding, medium confidence
        groups = [
            {"finding": item["claim"], "supporting_claim_ids": [i + 1],
             "contradicting_claim_ids": [], "note": ""}
            for i, item in enumerate(evidence_items)
        ]

    findings = []
    for g in groups:
        support_ids = [i for i in g.get("supporting_claim_ids", []) if 1 <= i <= len(evidence_items)]
        contra_ids = [i for i in g.get("contradicting_claim_ids", []) if 1 <= i <= len(evidence_items)]

        supporting = [evidence_items[i - 1] for i in support_ids]
        contradicting = [evidence_items[i - 1] for i in contra_ids]

        if not supporting:
            continue  # nothing to anchor this finding to, skip

        unique_sources = {e["source_url"] for e in supporting}
        avg_vgrh = sum(e.get("vgrh_score", 0.0) for e in supporting) / len(supporting)

        confidence = _confidence_label(
            num_sources=len(unique_sources),
            avg_vgrh=avg_vgrh,
            has_contradiction=len(contradicting) > 0,
        )

        findings.append({
            "finding": g.get("finding", "").strip(),
            "confidence": confidence,
            "num_sources": len(unique_sources),
            "avg_vgrh_score": round(avg_vgrh, 3),
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "note": g.get("note", "").strip(),
            "sub_question": sub_question,
        })

    return findings


if __name__ == "__main__":
    # Quick manual test
    sample_evidence = [
        {"claim": "OPEC+ extended production cuts of 2.2 million bpd through Q2 2026.",
         "evidence_snippet": "...", "source_url": "https://eia.gov/a", "source_title": "EIA",
         "vgrh_score": 0.8, "sub_question": "x"},
        {"claim": "OPEC+ has begun unwinding production cuts, increasing output.",
         "evidence_snippet": "...", "source_url": "https://oilprice.com/b", "source_title": "OilPrice",
         "vgrh_score": 0.6, "sub_question": "x"},
    ]
    findings = verify_subquestion_evidence("What are OPEC+ doing with production?", sample_evidence)
    for f in findings:
        print(f["confidence"], "-", f["finding"], "| sources:", f["num_sources"], "| note:", f["note"])

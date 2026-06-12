"""
planner.py
Layer 2 (Research Planner) + Layer 3 (Query Generator).

Takes the raw user query and:
  1. Decomposes it into a small set of sub-questions covering the
     key dimensions of the topic.
  2. For each sub-question, generates 1-2 concrete search engine
     queries (with synonyms/keywords) suitable for a search API.
"""

from llm import call_llm_json
from config import NUM_SUBQUESTIONS, SEARCH_QUERIES_PER_SUBQUESTION


PLANNER_PROMPT = """You are a research planning assistant.

The user wants a deep research report on the following question:

"{query}"

Break this down into {n} focused sub-questions that together cover the
important angles needed to answer the main question well (e.g. different
options/methods, costs, effectiveness, risks, evidence quality, etc.
depending on what's relevant to this specific topic).

For current affairs, geopolitics, military action, sanctions, conflict,
or public-policy questions, make the sub-questions cover:
- the immediate trigger or stated justification,
- the strategic/political context,
- consequences, escalation risks, and humanitarian impact,
- legal/international reaction and regional alliances.

For each sub-question, also produce 1-2 concrete web search queries
(short, keyword-style, the kind you'd type into a search engine) that
would help find sources to answer it.

Search-query rules:
- Include the current year when the query asks about recent or current events.
- Prefer authoritative or news/research terms such as official statement,
  Reuters, AP, BBC, Al Jazeera, CFR, CSIS, IISS, UN, or congressional report.
- Avoid social/video-first wording unless the user specifically asks for it.

Respond with ONLY a JSON array in this exact shape:
[
  {{
    "sub_question": "...",
    "search_queries": ["...", "..."]
  }},
  ...
]
"""


def plan_research(query: str, n: int = NUM_SUBQUESTIONS):
    """
    Returns a list of dicts:
      [{"sub_question": str, "search_queries": [str, ...]}, ...]
    """
    prompt = PLANNER_PROMPT.format(query=query, n=n)
    plan = call_llm_json(prompt, max_tokens=1024)

    # Basic validation / fallback
    if not isinstance(plan, list) or len(plan) == 0:
        raise ValueError(f"Planner returned unexpected format: {plan}")

    sanitized_plan = []
    for item in plan[:n]:
        if "sub_question" not in item or "search_queries" not in item:
            raise ValueError(f"Planner item missing fields: {item}")

        queries = [
            q.strip()
            for q in item["search_queries"]
            if isinstance(q, str) and q.strip()
        ][:SEARCH_QUERIES_PER_SUBQUESTION]
        if not queries:
            queries = [item["sub_question"]]

        sanitized_plan.append({
            "sub_question": item["sub_question"].strip(),
            "search_queries": queries,
        })

    return sanitized_plan


if __name__ == "__main__":
    # Quick manual test
    test_query = "Compare three low-cost methods for improving air quality in classrooms."
    plan = plan_research(test_query)
    for item in plan:
        print("\nSub-question:", item["sub_question"])
        for q in item["search_queries"]:
            print("   search:", q)

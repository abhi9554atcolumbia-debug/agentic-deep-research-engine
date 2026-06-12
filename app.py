"""
app.py
Streamlit UI for the Agentic Deep Research Engine.

Run with:
    streamlit run app.py

Shows:
  - Query input + iteration toggle
  - Live progress through the pipeline (planning, search, fetch,
    ranking, evidence, verification, refinement)
  - Per sub-question: VGRH-ranked source table + findings with
    confidence levels
  - Final cited report (rendered + downloadable as Markdown)
"""

import streamlit as st
from pipeline import run_pipeline_stream
from report_generator import calculate_research_metrics

st.set_page_config(page_title="Agentic Deep Research Engine", layout="wide")

st.title("🔎 Agentic Deep Research Engine")
st.caption("Decomposes your question, searches the web, ranks sources with VGRH, "
           "extracts & verifies evidence, and generates a cited report.")

# --- Input ---
with st.form("query_form"):
    query = st.text_area(
        "Research question",
        value="Why is the USA striking Iran?",
        height=80,
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        enable_iteration = st.checkbox("Enable iterative refinement", value=True,
                                        help="If a sub-question has no or low-confidence findings, "
                                             "run one extra targeted search round.")
    submitted = st.form_submit_button("Run Research")

if submitted and query.strip():
    plan_placeholder = st.empty()
    progress_log = st.expander("Pipeline progress log", expanded=False)
    subq_containers = {}  # sub_question -> streamlit container
    result = None

    plan = None
    findings_by_subq = {}
    ranked_by_subq = {}

    for event in run_pipeline_stream(query.strip(), enable_iteration=enable_iteration):
        etype = event["type"]

        if etype == "plan":
            plan = event["plan"]
            with plan_placeholder.container():
                st.subheader("Research Plan")
                for item in plan:
                    st.markdown(f"**{item['sub_question']}**")
                    st.caption("Search queries: " + ", ".join(f"`{q}`" for q in item["search_queries"]))
            # Create a results section per sub-question, in order
            st.subheader("Findings by Sub-question")
            for item in plan:
                subq_containers[item["sub_question"]] = st.container()

        elif etype == "search":
            progress_log.write(f"🔍 Search `{event['query']}` → {event['count']} results "
                                f"(for: *{event['sub_question'][:60]}...*)")

        elif etype == "fetch":
            progress_log.write(f"📄 Fetched {event['url']} → {event['n_chunks']} chunks")

        elif etype == "ranked":
            ranked_by_subq[event["sub_question"]] = event["chunks"]

        elif etype == "evidence":
            progress_log.write(f"🧩 Extracted {len(event['evidence'])} evidence items "
                                f"for: *{event['sub_question'][:60]}...*")

        elif etype == "refinement":
            progress_log.write(f"♻️ Refining sub-question with new queries: "
                                + ", ".join(f"`{q}`" for q in event["queries"]))

        elif etype == "findings":
            sub_q = event["sub_question"]
            findings = event["findings"]
            findings_by_subq[sub_q] = findings

            container = subq_containers.get(sub_q)
            if container is None:
                continue

            with container:
                st.markdown(f"### {sub_q}")

                # --- Findings ---
                if not findings:
                    st.warning("No verified findings for this sub-question.")
                else:
                    for f in findings:
                        badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(f["confidence"], "⚪")
                        st.markdown(f"{badge} **{f['confidence'].upper()}** "
                                    f"({f['num_sources']} source(s)) — {f['finding']}")
                        if f["note"]:
                            st.caption(f"⚠️ {f['note']}")

                # --- Source table for this sub-question ---
                chunks = ranked_by_subq.get(sub_q, [])
                if chunks:
                    with st.expander(f"Show ranked sources ({len(chunks)})"):
                        seen = set()
                        for c in chunks:
                            if c["url"] in seen:
                                continue
                            seen.add(c["url"])
                            b = c["vgrh_breakdown"]
                            st.markdown(
                                f"**[{c.get('title') or c['url']}]({c['url']})** — "
                                f"VGRH `{c['vgrh_score']:.2f}` "
                                f"(V:{b['veracity']:.2f} G:{b['grounding']:.2f} "
                                f"R:{b['relevance']:.2f} H:{b['helpfulness']:.2f})"
                            )
                st.divider()

        elif etype == "report":
            result = event

        elif etype == "done":
            result = event["result"]

        elif etype == "error":
            st.error(f"Pipeline error: {event['message']}")

    # --- Final report ---
    if result and "report_md" in result:
        metrics = calculate_research_metrics(
            result.get("plan", []),
            result.get("findings_by_subq", {}),
            result.get("all_ranked_chunks", []),
        )
        st.subheader("📋 Final Research Report")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sources Analyzed", metrics["sources_analyzed"])
        col2.metric("Evidence Items", metrics["evidence_items"])
        col3.metric("Verified Claims", metrics["verified_claims"])
        col4.metric("Research Confidence", f"{metrics['research_confidence']}%")
        st.markdown(result["report_md"])
        st.download_button(
            "Download report as Markdown",
            data=result["report_md"],
            file_name="research_report.md",
            mime="text/markdown",
        )

elif submitted:
    st.warning("Please enter a research question.")

# Agentic Deep Research Engine

Hackathon project for building an agentic research system that turns a
natural-language question into a traceable Markdown report.

The app decomposes a question into sub-questions, searches the web,
fetches and chunks sources, retrieves relevant passages with local
embeddings, ranks evidence with an explainable VGRH score, verifies
claims, and produces a cited report.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with:

```bash
GEMINI_API_KEY=...
TAVILY_API_KEY=...
```

## Run

```bash
streamlit run app.py
```

The Streamlit UI shows the research plan, live pipeline progress,
per-sub-question findings, ranked sources, and the final report.

## API Usage Controls

The project is tuned for free-tier API limits:

- Planner output is capped at 2 search queries per sub-question.
- VGRH ranking and evidence extraction share one Gemini call per
  sub-question.
- Final report prose is generated locally by default, saving 2 Gemini
  calls per run.
- To enable one optional Gemini call for more polished final prose, set:

```bash
USE_LLM_REPORT=true
```

## Architecture Map

| Layer | Module |
|---|---|
| User Query | `app.py` |
| Research Planner | `planner.py` |
| Query Generator | `planner.py` |
| Search / Source Discovery | `search.py` |
| Web + PDF Fetcher | `fetch_parse.py` |
| Parser + Chunker | `fetch_parse.py` |
| Retriever | `retriever.py` |
| VGRH Ranker | `score_and_extract.py`, `domain_trust.py` |
| Evidence Extractor | `score_and_extract.py` |
| Claim Verifier | `claim_verifier.py` |
| Report Generator | `report_generator.py`, `export_report.py` |
| Iterate / Refine Loop | `iteration.py` |

## Notes

- Embeddings run locally with `sentence-transformers`, so retrieval does
  not consume API quota.
- Some sites block automated fetching; in that case the system falls
  back to the Tavily search snippet and marks the source as
  `snippet-only`.
- Markdown export works directly in the UI. DOCX/PDF conversion helpers
  are included in `export_report.py`, but PDF export depends on local
  pandoc/PDF tooling.

## Screenshots
<img width="2560" height="16512" alt="screencapture-localhost-8502-2026-06-12-15_19_47" src="https://github.com/user-attachments/assets/7d2a5c35-edba-4b6d-a077-2c03ad1c70c7" />

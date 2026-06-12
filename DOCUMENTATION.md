# Agentic Deep Research Engine — Submission Documentation

This follows the required documentation structure: problem
understanding, architecture, modules implemented, setup steps,
APIs/tools used, limitations, screenshots.

## 1. Problem Understanding

The system takes a natural-language research question and produces a
traceable research report. Instead of asking an LLM to answer directly,
it decomposes the problem into focused sub-questions, searches for
external sources, retrieves relevant chunks, ranks them with an
explainable VGRH score, extracts evidence, verifies claims, and then
builds a final cited report.

The goal is to make deep research auditable: a user can see which
sub-questions were asked, which sources were discovered, why each source
was ranked highly, which evidence supported each finding, and where the
final answer came from.

## 2. Architecture

The system implements the full 11-layer pipeline from the brief:

| # | Layer | Module(s) | Status |
|---|---|---|---|
| 1 | User Query | `app.py` (Streamlit input) | ✅ |
| 2 | Research Planner | `planner.py` | ✅ |
| 3 | Query Generator | `planner.py` | ✅ |
| 4 | Search / Source Discovery | `search.py` (Tavily API) | ✅ |
| 5 | Web + PDF Fetcher | `fetch_parse.py` | ✅ |
| 6 | Parser + Chunker | `fetch_parse.py` | ✅ |
| 7 | Retriever | `retriever.py` (local embeddings, cosine similarity, source-diversity cap) | ✅ |
| 8 | VGRH Ranker | `score_and_extract.py` + `domain_trust.py` | ✅ |
| 9 | Evidence Extractor | `score_and_extract.py` (combined with VGRH for LLM-call efficiency) | ✅ |
| 10 | Claim Verifier | `claim_verifier.py` | ✅ |
| 11 | Report Generator | `report_generator.py` + `export_report.py` | ✅ |
| - | Iterate/Refine loop | `iteration.py` | ✅ |

Architecture diagram/screenshots can be added here before submission.

### VGRH Scoring (Layer 8) — explainability

Each retrieved chunk gets a score from 0-1 on four axes, combined with
weights defined in `config.py` (`VGRH_WEIGHTS`):

- **Veracity** = `0.6 * domain_trust_score(url)` (static lookup table
  in `domain_trust.py` — .gov/.edu/major institutions score highest)
  `+ 0.4 * LLM-judged content trustworthiness` (specificity, data,
  attribution vs. vague/promotional language).
- **Grounding** = lexical keyword overlap between the sub-question and
  the chunk text.
- **Relevance** = cosine similarity between sub-question and chunk
  embeddings (sentence-transformers, local, no API cost).
- **Helpfulness** = LLM-judged 1-5 rating of how specific/actionable
  the chunk's information is.

The full V/G/R/H breakdown is shown per source in the UI and in the
final report's source table — nothing is a black-box score.

### Confidence levels (Layer 10)

- **High**: ≥2 independent sources agree, average VGRH ≥ 0.55, no contradictions.
- **Medium**: at least 1 source, average VGRH ≥ 0.4, no contradictions.
- **Low**: contradicting sources found, or insufficient support.

Findings with no/low-confidence support trigger the iteration loop
(`iteration.py`), which generates 1-2 new targeted search queries and
runs one additional retrieval round.

## 3. Setup Instructions

```bash
git clone <your-repo-url>
cd research_engine
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in your API keys
streamlit run app.py
```

[Optional, for DOCX/PDF export] Install pandoc:
- macOS: `brew install pandoc` (+ `brew install basictex` for PDF)
- Linux: `sudo apt-get install pandoc texlive`

## 4. APIs / Tools Used

- **LLM**: Gemini 2.5 Flash via google-generativeai. The free-tier
  quota of 20 requests/day affected development, so the base pipeline
  is optimized to about 9 Gemini calls per full 4-sub-question run
  before optional refinement.
- **Search**: Tavily Search API (free tier)
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (local, no API cost)
- **Parsing**: trafilatura (HTML), pypdf (PDF)
- **UI**: Streamlit
- **Export**: pypandoc / pandoc (DOCX, PDF)

API-call optimizations:

- Planner search queries are capped at 2 per sub-question to reduce
  search calls.
- VGRH ranking and evidence extraction are combined into 1 Gemini call
  per sub-question.
- Report prose is generated locally by default, saving 2 Gemini calls
  per full run. Setting `USE_LLM_REPORT=true` enables 1 optional Gemini
  call for a more polished executive summary and synthesis.

## 5. Sample Input / Output

**Sample query**: "What is the reason for oil price rising?"

Attach `output_report.md` or a screenshot from the Streamlit final
report section.

## 6. Limitations

- Many high-authority sources (IEA, ScienceDirect, Reuters, WSJ, etc.)
  block automated fetching (403/401 errors); the system falls back to
  using the search-result snippet for these, which provides less
  detail than the full article.
- VGRH veracity scoring uses a static domain-trust lookup table
  (`domain_trust.py`) that covers common energy/finance/research
  domains — unrecognized domains get a neutral default score.
- The iteration/refinement loop is capped at 1 round per sub-question
  to manage LLM API rate limits (especially relevant on free-tier
  LLM APIs).
- PDF export requires a local LaTeX/PDF engine in addition to pandoc;
  if unavailable, Markdown and DOCX exports still work.
- Final prose is intentionally extractive by default to stay within
  free-tier API limits; enabling `USE_LLM_REPORT=true` improves prose
  polish at the cost of one extra Gemini call.

## 7. Screenshots

Add screenshots of: the Streamlit input form, live progress log, a
sub-question's findings with confidence badges, the source table with
VGRH breakdown, and the final report.

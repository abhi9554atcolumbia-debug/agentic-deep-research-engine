"""
config.py
Central configuration: API keys, model names, pipeline constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM model used for planning, query generation, evidence extraction, etc.
LLM_MODEL = "gemini-2.5-flash"

# Embedding model (local, free, no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Pipeline tunables
NUM_SUBQUESTIONS = 4          # how many sub-questions to decompose into
SEARCH_QUERIES_PER_SUBQUESTION = 2  # cap planner output to control search API usage
RESULTS_PER_SUBQUESTION = 4   # how many search results to fetch per sub-question
CHUNK_SIZE = 800              # characters per chunk
CHUNK_OVERLAP = 150           # overlap between chunks
TOP_K_RETRIEVAL = 8           # how many chunks to retrieve per sub-question
USE_LLM_REPORT = os.getenv("USE_LLM_REPORT", "false").lower() == "true"

# Keep low-signal/social pages out of the default research set. This
# improves claim extraction for serious policy/current-affairs queries.
EXCLUDED_SOURCE_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "reddit.com",
}

# --- Phase 2 tunables ---

# VGRH weighted scoring (must sum to 1.0)
VGRH_WEIGHTS = {
    "veracity": 0.30,
    "grounding": 0.20,
    "relevance": 0.25,
    "helpfulness": 0.25,
}

# Veracity = blend of domain-trust lookup table and LLM content judgment
VERACITY_DOMAIN_WEIGHT = 0.6
VERACITY_LLM_WEIGHT = 0.4

# How many top-VGRH chunks per sub-question go to evidence extraction
TOP_N_FOR_EVIDENCE = 8

# Minimum number of independent sources for a claim to be "high confidence"
MIN_SOURCES_HIGH_CONFIDENCE = 2

TRUSTED_DOMAINS = {
    "eia.gov": 0.05,
    "imf.org": 0.05,
    "worldbank.org": 0.05,
    "opec.org": 0.05,
    "iea.org": 0.05,
    "ecb.europa.eu": 0.05,
    "oxfordenergy.org": 0.05,
    "pmc.ncbi.nlm.nih.gov": 0.05,
}

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Set it in .env")
if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not set. Set it in .env")

"""
search.py
Layer 4 (Search / Source Discovery).

Wraps the Tavily search API to discover candidate sources
(web pages, articles, PDFs) for a given search query.
"""

from tavily import TavilyClient
from urllib.parse import urlparse
from config import TAVILY_API_KEY, RESULTS_PER_SUBQUESTION, EXCLUDED_SOURCE_DOMAINS

_client = TavilyClient(api_key=TAVILY_API_KEY)


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _is_excluded(url: str) -> bool:
    host = _domain(url)
    return any(host == domain or host.endswith(f".{domain}") for domain in EXCLUDED_SOURCE_DOMAINS)


def search_sources(query: str, max_results: int = RESULTS_PER_SUBQUESTION):
    """
    Run a web search and return a list of candidate sources:
      [{"url": str, "title": str, "snippet": str}, ...]
    """
    try:
        response = _client.search(
            query=query,
            max_results=max_results + len(EXCLUDED_SOURCE_DOMAINS),
            search_depth="basic",  # use "advanced" for deeper but slower/costlier search
        )
    except Exception as e:
        print(f"  [search error] '{query}': {e}")
        return []

    results = []
    for r in response.get("results", []):
        url = r.get("url")
        if not url or _is_excluded(url):
            continue
        results.append({
            "url": url,
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
        })
        if len(results) >= max_results:
            break
    return results


if __name__ == "__main__":
    # Quick manual test
    results = search_sources("low-cost classroom air purifier methods")
    for r in results:
        print(r["url"], "-", r["title"])

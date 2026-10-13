import logging
from typing import Any
from tavily import TavilyClient
from app.core.config import get_settings

logger = logging.getLogger(__name__)


_tavily_client: Any = None


def get_tavily_client() -> Any:
    """Singleton getter for TavilyClient instance."""
    global _tavily_client
    settings = get_settings()
    if _tavily_client is None:
        if not settings.tavily_api_key:
            return None
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


def execute_tavily_search(
    query: str,
    max_results: int = 4,
    tavily_client: Any = None,
) -> list[dict[str, Any]]:
    """
    Execute web search via Tavily Search API.
    Preserves title, url, snippet/content, and handles errors cleanly.
    Reuses initialized client and expands results for full-year calendar queries.
    """
    settings = get_settings()
    if not settings.tavily_api_key and tavily_client is None:
        logger.warning("TAVILY_API_KEY is not configured; web search fallback unavailable.")
        return []

    client = tavily_client or get_tavily_client()
    if client is None:
        return []

    # For full-year holiday or calendar queries, request more results for full-year coverage
    effective_max = max_results
    q_lower = query.lower()
    if any(k in q_lower for k in ["holiday", "holidays", "calendar", "gazetted"]):
        effective_max = max(max_results, 6)

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=effective_max,
            include_answer=True,
        )
    except Exception as exc:
        logger.error("Tavily search API failed for query '%s': %s", query, exc)
        return []

    results: list[dict[str, Any]] = []

    # If Tavily returned an AI answer snippet, append as a synthesized summary
    if isinstance(response, dict):
        ai_summary = response.get("answer")
        raw_items = response.get("results", [])

        for item in raw_items:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").strip()
            if url:
                results.append({
                    "title": title or url,
                    "url": url,
                    "content": content,
                })

        if ai_summary and results:
            results[0]["ai_summary"] = ai_summary

    return results

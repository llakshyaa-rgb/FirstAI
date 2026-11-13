"""Read-only Brave web search tool for Orbit."""
from __future__ import annotations
import json
import os
import urllib.parse
import urllib.request

def search_web(query: str, count: int = 8) -> list[dict]:
    query = str(query or "").strip()
    if not query or len(query) > 400:
        raise ValueError("Enter a web search query up to 400 characters.")
    token = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not token:
        raise RuntimeError("Web search is not configured. Set BRAVE_SEARCH_API_KEY.")
    params = urllib.parse.urlencode({"q": query, "count": max(1, min(int(count), 10)), "safesearch": "moderate"})
    request = urllib.request.Request(
        "https://api.search.brave.com/res/v1/web/search?" + params,
        headers={"Accept": "application/json", "X-Subscription-Token": token, "User-Agent": "OrbitLocal/0.2"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read(4_000_000).decode("utf-8"))
    results = []
    for item in (payload.get("web") or {}).get("results", []):
        url = item.get("url")
        if not url:
            continue
        results.append({
            "title": str(item.get("title") or "Search result")[:240],
            "url": url,
            "description": str(item.get("description") or "")[:1000],
            "source": str(item.get("meta_url", {}).get("hostname") or ""),
        })
    return results

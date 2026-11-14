"""Source-attributed, read-only knowledge retrieval for Orbit.

Uses documented public APIs for encyclopedic, scholarly, and programming
knowledge. User text selects a search term; it never becomes an arbitrary URL.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

TIMEOUT = 18
USER_AGENT = "FirstAI-Orbit/1.0 (source-attributed knowledge retrieval)"
MAX_RESULTS = 5


def _get(url, accept="application/json"):
    request = urllib.request.Request(url, headers={
        "Accept": accept, "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Knowledge source response exceeded 2 MB.")
    return raw.decode("utf-8")


def _search_term(query):
    value = re.sub(r"^\s*(?:research|look up|find sources for|explain|what is|who is|tell me about)\s+", "", query, flags=re.I)
    return value.strip(" \t\r\n?!.")[:240]


def _wikipedia(query):
    term = _search_term(query)
    if not term:
        raise ValueError("Provide a topic to look up.")
    url = "https://en.wikipedia.org/w/rest.php/v1/search/page?" + urllib.parse.urlencode({
        "q": term, "limit": MAX_RESULTS,
    })
    payload = json.loads(_get(url))
    rows = []
    for page in (payload.get("pages") or [])[:MAX_RESULTS]:
        title = page.get("title")
        if not title:
            continue
        rows.append({
            "title": title,
            "description": page.get("description"),
            "excerpt": re.sub(r"<[^>]+>", "", page.get("excerpt") or "")[:900],
            "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
        })
    return {
        "domain": "general_knowledge",
        "source": "Wikipedia",
        "source_url": "https://www.mediawiki.org/wiki/API:REST_API",
        "license": "Wikipedia text is generally CC BY-SA; attribute the article and follow share-alike terms.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": term, "records": rows,
        "caveats": ["Search excerpts are discovery material, not a substitute for checking the cited article and its references."],
    }


def _openalex(query):
    term = _search_term(query)
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": term, "per-page": MAX_RESULTS,
        "select": "id,doi,title,publication_year,type,cited_by_count,authorships,primary_location,open_access",
    })
    payload = json.loads(_get(url))
    rows = []
    for item in payload.get("results", [])[:MAX_RESULTS]:
        authors = []
        for author in (item.get("authorships") or [])[:5]:
            name = (author.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        rows.append({
            "title": item.get("title"),
            "year": item.get("publication_year"),
            "type": item.get("type"),
            "authors": authors,
            "doi": item.get("doi"),
            "cited_by_count": item.get("cited_by_count"),
            "venue": source.get("display_name"),
            "open_access": (item.get("open_access") or {}).get("is_oa"),
            "landing_page_url": location.get("landing_page_url"),
            "open_access_url": (item.get("open_access") or {}).get("oa_url"),
            "openalex_id": item.get("id"),
        })
    return {
        "domain": "scholarly_research",
        "source": "OpenAlex",
        "source_url": "https://docs.openalex.org/",
        "license": "OpenAlex bibliographic metadata is CC0; article abstracts and full text have separate rights.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": term, "records": rows,
        "caveats": ["A paper's presence or citation count does not establish that its claims are correct. Check methods, publication status and the original paper."],
    }


def _arxiv(query):
    term = _search_term(query)
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": "all:" + term, "start": 0, "max_results": MAX_RESULTS,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    root = ET.fromstring(_get(url, "application/atom+xml"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("a:entry", ns):
        rows.append({
            "title": " ".join((entry.findtext("a:title", default="", namespaces=ns)).split()),
            "summary": " ".join((entry.findtext("a:summary", default="", namespaces=ns)).split())[:1100],
            "published": entry.findtext("a:published", namespaces=ns),
            "updated": entry.findtext("a:updated", namespaces=ns),
            "authors": [x.text for x in entry.findall("a:author/a:name", ns)[:6]],
            "url": entry.findtext("a:id", namespaces=ns),
            "categories": [x.attrib.get("term") for x in entry.findall("a:category", ns)[:5]],
        })
    return {
        "domain": "scholarly_research",
        "source": "arXiv",
        "source_url": "https://info.arxiv.org/help/api/index.html",
        "license": "Metadata is searchable; article and source-code licenses vary by paper.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": term, "records": rows,
        "caveats": ["arXiv includes preprints and papers that may not be peer reviewed. Verify publication status and article license."],
    }


def _stackoverflow(query):
    term = _search_term(query)
    url = "https://api.stackexchange.com/2.3/search/advanced?" + urllib.parse.urlencode({
        "site": "stackoverflow", "q": term, "pagesize": MAX_RESULTS,
        "order": "desc", "sort": "relevance", "filter": "default",
    })
    payload = json.loads(_get(url))
    rows = [{
        "title": item.get("title"),
        "score": item.get("score"),
        "answer_count": item.get("answer_count"),
        "is_answered": item.get("is_answered"),
        "creation_date_unix": item.get("creation_date"),
        "link": item.get("link"),
    } for item in payload.get("items", [])[:MAX_RESULTS]]
    return {
        "domain": "programming_knowledge",
        "source": "Stack Exchange API (Stack Overflow)",
        "source_url": "https://api.stackexchange.com/docs",
        "license": "User contributions are generally CC BY-SA; inspect post revision and attribution requirements before reuse.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": term, "records": rows,
        "caveats": ["Search metadata is not a verified solution. Open answers, inspect dates and versions, and attribute user contributions."],
        "quota_remaining": payload.get("quota_remaining"),
    }


def search_knowledge(query: str) -> dict:
    """Search one or more trusted knowledge APIs based on the requested domain."""
    text = str(query or "").strip()
    if not text or len(text) > 500:
        raise ValueError("Enter a knowledge query up to 500 characters.")
    lower = text.casefold()
    if any(word in lower for word in ("stackoverflow", "python error", "javascript error", "programming", "code example", "coding question", "api usage")):
        results = [_stackoverflow(text)]
    elif any(word in lower for word in ("arxiv", "preprint", "scientific paper", "research paper", "academic paper")):
        results = [_arxiv(text)]
    elif any(word in lower for word in ("research", "literature review", "papers about", "studies on", "academic sources")):
        results = [_openalex(text), _arxiv(text)]
    else:
        results = [_wikipedia(text)]
    for result in results:
        result["assistant_context"] = (
            f"{result['domain']} via {result['source']}; retrieved {result['retrieved_at_utc']}. "
            + "Results: " + json.dumps(result["records"][:MAX_RESULTS], ensure_ascii=False)[:3800]
            + f". Source: {result['source_url']}. License: {result['license']}. "
            + "Caveats: " + " ".join(result.get("caveats", []))
        )[:5000]
    return {"query": text, "results": results,
            "assistant_context": "\n\n".join(x["assistant_context"] for x in results)[:9000]}

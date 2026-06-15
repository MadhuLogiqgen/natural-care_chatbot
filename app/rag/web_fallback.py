from ddgs import DDGS

from app.config import WEB_SEARCH_MAX_RESULTS
from app.models import Source


def search_web(question: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list[Source]:
    """Search the web for natural face and hair care information."""
    query = f"natural face hair care {question}"
    sources: list[Source] = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for result in results:
                title = result.get("title") or "Web result"
                body = result.get("body") or ""
                url = result.get("href") or ""
                if not body and not title:
                    continue
                sources.append(
                    Source(
                        filename=title,
                        page=0,
                        excerpt=body[:400] + ("..." if len(body) > 400 else ""),
                        source_type="web",
                        url=url or None,
                        title=title,
                    )
                )
    except Exception:
        return []

    return sources

from __future__ import annotations

import json
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class CrossrefClient:
    """Small read-only Crossref client for citation discovery and DOI verification."""

    base_url = "https://api.crossref.org"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search(self, query: str, rows: int = 8) -> list[dict]:
        if not query.strip():
            return []
        params = urlencode({"query.bibliographic": query.strip(), "rows": min(max(rows, 1), 20), "select": "DOI,title,author,published,container-title,URL,type"})
        data = self._get(f"/works?{params}")
        return [self._normalize(item) for item in data.get("message", {}).get("items", [])]

    def resolve(self, doi: str) -> dict:
        data = self._get(f"/works/{quote(doi.strip(), safe='')}")
        return self._normalize(data.get("message", {}))

    def _get(self, path: str) -> dict:
        request = Request(
            self.base_url + path,
            headers={"User-Agent": "PaperAI/2.0 (https://github.com/Lava168/PaperAI)", "Accept": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _normalize(item: dict) -> dict:
        date_parts = item.get("published", {}).get("date-parts", [[]])
        authors = []
        for author in item.get("author", []):
            name = " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part)
            if name:
                authors.append(name)
        return {
            "doi": item.get("DOI", ""),
            "title": (item.get("title") or [""])[0],
            "authors": authors,
            "year": date_parts[0][0] if date_parts and date_parts[0] else None,
            "venue": (item.get("container-title") or [""])[0],
            "url": item.get("URL", ""),
            "type": item.get("type", ""),
        }

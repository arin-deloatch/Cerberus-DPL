"""
Minimal WebAdapter (MVP):
- GET the URL
- Parse HTML
- Extract title, visible text, and outlinks
- Return a NormalizedDoc

Deliberately simple: HTML only, basic error handling, no language detection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

import httpx
from lxml import html as lhtml

from cerberus_dpl.models import NormalizedDoc
from cerberus_dpl.config import settings
from cerberus_dpl.logging import logger
from cerberus_dpl.adapters.base import (
    Adapter,
    AdapterError,
    NotFoundError,
    TransientError,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _strip_noise(tree: lhtml.HtmlElement) -> None:
    # Remove script and style elements
    for element in tree.xpath("//script | //style | //header | //nav | //footer"):
        element.getparent().remove(element)


def _extract_outlinks(tree: lhtml.HtmlElement, base_url: str) -> list[str]:
    hrefs = tree.xpath("//a[@href]/@href")
    out: list[str] = []
    seen: set[str] = set()
    for h in hrefs:
        if not h or h.startswith(("#", "mailto:", "javascript:")):
            continue
        u = urljoin(base_url, h)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


class WebAdapter(Adapter):
    name = "web"

    def __init__(
        self, timeout_s: Optional[float] = None, user_agent: Optional[str] = None
    ):
        self.timeout_s = timeout_s or float(settings.TIMEOUT_S)
        self.user_agent = user_agent or settings.USER_AGENT
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_s,
            follow_redirects=True,
        )

    def fetch_and_normalize(self, handle: str) -> NormalizedDoc:
        url = handle.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise AdapterError(f"Expected absolute http(s) URL, got: {url!r}")

        log = logger.bind(adapter=self.name, url=url)
        log.info("web_fetch_start")

        try:
            resp = self._client.get(url)
        except httpx.TimeoutException as e:
            log.warning("web_timeout", error=str(e))
            raise TransientError(f"Timeout fetching {url}") from e
        except httpx.HTTPError as e:
            log.warning("web_http_error", error=str(e))
            raise TransientError(f"HTTP error fetching {url}") from e

        status = resp.status_code
        ctype = resp.headers.get("content-type", "")
        final_url = str(resp.url)

        if status in (404, 410):
            log.info("web_not_found", status=status)
            raise NotFoundError(f"{status} for {url}")

        if status >= 500:
            log.warning("web_server_error", status=status)
            raise TransientError(f"Server error {status} for {url}")

        if "text/html" not in ctype.lower():
            log.info("web_unsupported_type", content_type=ctype)
            raise AdapterError(f"Unsupported content-type: {ctype}")

        raw_bytes = resp.content
        try:
            tree = lhtml.fromstring(raw_bytes)
        except Exception as e:
            log.error("web_parse_error", error=str(e))
            raise AdapterError(f"HTML parse failed for {final_url}") from e

        _strip_noise(tree)
        title_nodes = tree.xpath("//title/text()")
        title = (title_nodes[0].strip() if title_nodes else None) or None
        text = (tree.text_content() or "").strip()
        outlinks = _extract_outlinks(tree, final_url)

        ndoc = NormalizedDoc(
            source_type="web",
            source_id=final_url,  # use resolved URL as id
            timestamp=_utc_now(),
            uri=final_url,
            version_tag=resp.headers.get("etag"),
            license=None,
            mime_type="text/html",
            lang=None,  # MVP: skip detection
            title=title,
            text='text',
            outlinks=[],
            backlinks=[],
            headings=[],  # keep minimal
            code_blocks=0,
            tables=0,
            last_modified=None,  # MVP: omit; can parse later
            meta={
                "http_status": str(status),
                "content_type": ctype,
            },
        )

        log.info(
            "web_fetch_ok", status=status, text_len=len(text), outlinks=len(outlinks)
        )
        print(ndoc)
        return ndoc


if __name__ == "__main__":
    web_adapter = WebAdapter()

    web_adapter.fetch_and_normalize(
        "https://www.redhat.com/en/blog/openshift-vision-and-execution"
    )

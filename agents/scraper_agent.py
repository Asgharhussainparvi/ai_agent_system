"""
agents/scraper_agent.py
━━━━━━━━━━━━━━━━━━━━━━━
ScraperAgent — autonomously discovers trending AI/ML topics from:
  • HuggingFace Spaces & Papers (trending section)
  • DevPost (AI/ML hackathon projects)

Returns a ranked list of Topic objects ready for the ContentAgent.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from utils.config import HF_SCRAPE_LIMIT, DEVPOST_SCRAPE_LIMIT
from utils.logger import get_logger
from utils.retry import with_retry

_log = get_logger("ScraperAgent")

# ── Headers that look like a real browser ────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Topic:
    title: str
    source: str                  # "huggingface" | "devpost"
    url: str
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    score: float = 0.0           # Heuristic relevance score
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def as_pipeline_topic(self) -> str:
        """Return a rich string for ContentAgent consumption."""
        tag_str = ", ".join(self.tags[:5]) if self.tags else ""
        return (
            f"{self.title}"
            + (f" ({tag_str})" if tag_str else "")
            + (f" — {self.summary[:120]}" if self.summary else "")
        )


# ── HuggingFace Scraper ───────────────────────────────────────────────────────
class _HuggingFaceScraper:
    """
    Scrapes two HuggingFace pages:
      1. /spaces  — trending Spaces (community apps)
      2. /papers  — daily paper digest
    """

    BASE = "https://huggingface.co"
    SPACES_URL = "https://huggingface.co/spaces?sort=trending"
    PAPERS_URL = "https://huggingface.co/papers"

    @with_retry(max_retries=3, delay=3, exceptions=(requests.RequestException,))
    def _fetch(self, url: str) -> BeautifulSoup:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    def _scrape_spaces(self, limit: int) -> List[Topic]:
        _log.info("HF: scraping trending Spaces…")
        topics: List[Topic] = []
        try:
            soup = self._fetch(self.SPACES_URL)

            # Each space card has an <article> or a link with /spaces/ in href
            cards = soup.select("article") or soup.select("div[class*='SpaceCard']")

            # Fallback: grab all links pointing to /spaces/<user>/<name>
            if not cards:
                links = soup.find_all("a", href=re.compile(r"^/spaces/[^/]+/[^/]+$"))
                for link in links[:limit]:
                    title = link.get_text(strip=True) or link["href"].split("/")[-1]
                    topics.append(Topic(
                        title=title,
                        source="huggingface",
                        url=urljoin(self.BASE, link["href"]),
                        summary="Trending HuggingFace Space",
                        tags=["AI", "ML", "HuggingFace"],
                        score=0.8,
                    ))
                return topics[:limit]

            for card in cards[:limit]:
                link = card.find("a", href=True)
                if not link:
                    continue
                href = link.get("href", "")
                if not href.startswith("/spaces/"):
                    continue
                title = card.get_text(separator=" ", strip=True)[:120]
                title = re.sub(r"\s+", " ", title).strip()
                topics.append(Topic(
                    title=title or href.split("/")[-1],
                    source="huggingface",
                    url=urljoin(self.BASE, href),
                    summary="Trending HuggingFace Space",
                    tags=["AI", "ML", "HuggingFace", "Space"],
                    score=0.85,
                ))
        except Exception as exc:
            _log.warning("HF Spaces scrape failed: %s", exc)
        return topics[:limit]

    def _scrape_papers(self, limit: int) -> List[Topic]:
        _log.info("HF: scraping daily papers…")
        topics: List[Topic] = []
        try:
            soup = self._fetch(self.PAPERS_URL)

            # Paper titles are usually in <h3> or <h2> tags inside paper cards
            paper_links = soup.find_all(
                "a", href=re.compile(r"^/papers/\d{4}\.\d+")
            )

            seen = set()
            for link in paper_links:
                href = link.get("href", "")
                if href in seen:
                    continue
                seen.add(href)

                title = link.get_text(strip=True)
                if not title or len(title) < 10:
                    continue

                # Try to grab abstract from sibling elements
                parent = link.find_parent(["article", "div", "li"])
                summary = ""
                if parent:
                    p_tags = parent.find_all("p")
                    for p in p_tags:
                        text = p.get_text(strip=True)
                        if len(text) > 40:
                            summary = text[:200]
                            break

                topics.append(Topic(
                    title=title[:120],
                    source="huggingface",
                    url=urljoin(self.BASE, href),
                    summary=summary,
                    tags=["AI", "ML", "Research", "Paper", "HuggingFace"],
                    score=0.9,
                ))
                if len(topics) >= limit:
                    break
        except Exception as exc:
            _log.warning("HF Papers scrape failed: %s", exc)
        return topics[:limit]

    def scrape(self, limit: int = 10) -> List[Topic]:
        half = max(1, limit // 2)
        spaces = self._scrape_spaces(half)
        papers = self._scrape_papers(limit - len(spaces))
        combined = spaces + papers
        _log.info("HuggingFace: collected %d topics.", len(combined))
        return combined[:limit]


# ── DevPost Scraper ───────────────────────────────────────────────────────────
class _DevPostScraper:
    """
    Scrapes DevPost hackathon projects tagged with AI/ML.
    Uses the public search endpoint (no auth needed).
    """

    SEARCH_URL = "https://devpost.com/software/search"
    FEATURED_URL = "https://devpost.com/software/newest"

    _AI_QUERIES = ["artificial intelligence", "machine learning", "LLM", "generative AI"]

    @with_retry(max_retries=3, delay=3, exceptions=(requests.RequestException,))
    def _fetch(self, url: str, params: dict | None = None) -> BeautifulSoup:
        resp = requests.get(url, headers=_HEADERS, params=params or {}, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    def _parse_project_cards(self, soup: BeautifulSoup) -> List[dict]:
        """Parse software listing cards common to both search and browse pages."""
        projects = []
        # DevPost uses <div class="software-entry"> or <article>
        cards = (
            soup.select("div.software-entry")
            or soup.select("div[id^='software-']")
            or soup.select("article.software")
            or soup.select("div.gallery-item")
        )

        for card in cards:
            # Title
            title_el = (
                card.find("h5")
                or card.find("h3")
                or card.find("a", class_=re.compile(r"title|name|software"))
            )
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue

            # URL
            link = card.find("a", href=re.compile(r"devpost\.com/software/|^/software/"))
            url = ""
            if link:
                href = link.get("href", "")
                url = href if href.startswith("http") else f"https://devpost.com{href}"

            # Summary / tagline
            tagline_el = card.find("p", class_=re.compile(r"tagline|description|excerpt"))
            summary = tagline_el.get_text(strip=True)[:200] if tagline_el else ""

            # Tags
            tag_els = card.find_all("span", class_=re.compile(r"tag|tech|built"))
            tags = [t.get_text(strip=True) for t in tag_els if t.get_text(strip=True)]
            tags = list(dict.fromkeys(tags))[:8]   # deduplicate

            projects.append({
                "title": title[:120],
                "url": url,
                "summary": summary,
                "tags": tags or ["AI", "ML", "DevPost"],
            })
        return projects

    def _scrape_search(self, query: str, limit: int) -> List[Topic]:
        topics: List[Topic] = []
        try:
            soup = self._fetch(self.SEARCH_URL, {"q": query})
            projects = self._parse_project_cards(soup)
            for p in projects[:limit]:
                topics.append(Topic(
                    title=p["title"],
                    source="devpost",
                    url=p["url"],
                    summary=p["summary"],
                    tags=p["tags"],
                    score=0.75,
                ))
        except Exception as exc:
            _log.warning("DevPost search '%s' failed: %s", query, exc)
        return topics

    def _scrape_featured(self, limit: int) -> List[Topic]:
        """Fallback: scrape newest/featured projects page."""
        topics: List[Topic] = []
        try:
            soup = self._fetch(self.FEATURED_URL)
            projects = self._parse_project_cards(soup)
            for p in projects[:limit]:
                topics.append(Topic(
                    title=p["title"],
                    source="devpost",
                    url=p["url"],
                    summary=p["summary"],
                    tags=p["tags"],
                    score=0.65,
                ))
        except Exception as exc:
            _log.warning("DevPost featured scrape failed: %s", exc)
        return topics

    def scrape(self, limit: int = 10) -> List[Topic]:
        topics: List[Topic] = []
        per_query = max(2, limit // len(self._AI_QUERIES))

        for query in self._AI_QUERIES:
            _log.info("DevPost: searching '%s'…", query)
            batch = self._scrape_search(query, per_query)
            for t in batch:
                if not any(x.title == t.title for x in topics):
                    topics.append(t)
            if len(topics) >= limit:
                break
            time.sleep(1)   # polite delay between queries

        if not topics:
            _log.warning("DevPost search yielded nothing — trying featured page…")
            topics = self._scrape_featured(limit)

        _log.info("DevPost: collected %d topics.", len(topics))
        return topics[:limit]


# ── ScraperAgent (public) ─────────────────────────────────────────────────────
class ScraperAgent:
    """
    Autonomous topic-discovery agent.

    Usage:
        agent  = ScraperAgent()
        topics = agent.run()          # returns List[Topic]
        best   = topics[0].as_pipeline_topic()   # feed to ContentAgent
    """

    def __init__(self):
        self._hf      = _HuggingFaceScraper()
        self._devpost = _DevPostScraper()

    def run(
        self,
        hf_limit: int = HF_SCRAPE_LIMIT,
        devpost_limit: int = DEVPOST_SCRAPE_LIMIT,
    ) -> List[Topic]:
        _log.info("ScraperAgent: starting topic discovery…")

        hf_topics      = self._hf.scrape(hf_limit)
        devpost_topics = self._devpost.scrape(devpost_limit)

        all_topics = hf_topics + devpost_topics

        # Rank by score (desc), then by title length (shorter = cleaner)
        all_topics.sort(key=lambda t: (-t.score, len(t.title)))

        _log.info(
            "ScraperAgent ✓  %d topics total (HF: %d, DevPost: %d)",
            len(all_topics), len(hf_topics), len(devpost_topics),
        )
        return all_topics

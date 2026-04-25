#!/usr/bin/env python3
"""
Main ingestion script — fetches from all sources, classifies, and writes to raw/.

Usage:
  python scripts/ingest.py

Environment variables (all optional unless noted):
  OPENAI_API_KEY      — GPT-4o-mini for classification (required in CI)
  ANTHROPIC_API_KEY   — Claude Haiku fallback
  ZHIPU_API_KEY       — GLM fallback
  JINA_API_KEY        — Jina Reader (optional, falls back to direct fetch)
  X_BEARER_TOKEN      — X API v2 Bearer token
  X_ACCESS_TOKEN       — X OAuth 1.0a access token
  GH_PAT              — GitHub PAT (required in CI)
  GH_REPO             — repo slug "owner/name"
  LOG_LEVEL           — DEBUG, INFO (default INFO)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

# ─── Logging ────────────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"ingest_{datetime.date.today()}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ingest")

# ─── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.resolve()
RAW_DIR = REPO_ROOT / "raw"
DATA_DIR = REPO_ROOT / "data"
CATEGORIES = ["ai-ml", "coding", "design", "productivity", "research", "business", "other"]

for cat in CATEGORIES:
    (RAW_DIR / cat).mkdir(parents=True, exist_ok=True)

# Ensure lowercase category dirs exist (migrate from old "AI-ML" naming)
(RAW_DIR / "AI-ML").mkdir(parents=True, exist_ok=True)  # legacy

(DATA_DIR).mkdir(parents=True, exist_ok=True)
DATA_DIR.joinpath(".gitkeep").touch()

# ─── LLM Provider ───────────────────────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def llm_classify(text: str, url: str) -> dict:
    """
    Classify + score + summarize a piece of content.
    Returns dict: {category, score, summary, tags}
    """
    SYSTEM = """你是一个知识库管理员。根据内容主题，将其分类到以下类别之一：
- AI-ML（人工智能、机器学习相关）
- coding（编程、开发相关）
- design（设计、UI/UX相关）
- productivity（效率工具、工作流相关）
- research（学术论文、研究相关）
- business（商业、创业、投资相关）
- other（不属于以上类别）

输出格式（纯JSON，无markdown）：
{
  "category": "类别名（小写，用-连接，如ai-ml）",
  "score": 1-10（内容质量/有用程度打分）,
  "summary": "一句话摘要（中文，50字以内）",
  "tags": ["标签1", "标签2", "标签3"]
}"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"URL: {url}\n\n内容前3000字：\n{text[:3000]}",
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return _llm_openai(payload, OPENAI_API_KEY)
    elif LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        return _llm_anthropic(text[:3000], ANTHROPIC_API_KEY)
    elif LLM_PROVIDER == "zhipu" and ZHIPU_API_KEY:
        return _llm_zhipu(text[:3000], ZHIPU_API_KEY)
    elif LLM_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
        return _llm_deepseek(payload, DEEPSEEK_API_KEY)
    elif LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return _llm_groq(payload, GROQ_API_KEY)
    elif LLM_PROVIDER == "ollama":
        return _llm_ollama(payload)
    else:
        log.warning("No LLM API key — using rule-based fallback classification")
        return _rule_classify(url, text[:500])


def _llm_openai(payload: dict, api_key: str) -> dict:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _llm_anthropic(text: str, api_key: str) -> dict:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-2025-05-14",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": f"Classify this content. Return JSON: {text[:2000]}",
                }
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["content"][0]["text"])


def _llm_zhipu(text: str, api_key: str) -> dict:
    resp = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4-flash",
            "messages": [
                {
                    "role": "user",
                    "content": f"根据以下内容返回JSON分类结果：{text[:2000]}",
                }
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _llm_deepseek(payload: dict, api_key: str) -> dict:
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={**payload, "model": "deepseek-chat"},
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _llm_groq(payload: dict, api_key: str) -> dict:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={**payload, "model": "llama-3.3-70b-versatile"},
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _llm_ollama(payload: dict) -> dict:
    """Ollama local model — OpenAI-compatible endpoint."""
    # Ollama doesn't support response_format
    payload = {k: v for k, v in payload.items() if k != "response_format"}
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json={**payload, "model": OLLAMA_MODEL},
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _rule_classify(url: str, text: str) -> dict:
    """Fallback when no LLM is available — rule-based classification."""
    text_lower = (url + " " + text).lower()
    if any(k in text_lower for k in ["ai", "llm", "gpt", "transformer", "neural", "diffusion"]):
        cat = "ai-ml"
    elif any(k in text_lower for k in ["python", "javascript", "code", "github", "api", "programming"]):
        cat = "coding"
    elif any(k in text_lower for k in ["design", "figma", "ui", "ux", "interface"]):
        cat = "design"
    elif any(k in text_lower for k in ["productivity", "workflow", "notion", "obsidian", "automation"]):
        cat = "productivity"
    elif any(k in text_lower for k in ["paper", "research", "arxiv", "study", "论文"]):
        cat = "research"
    elif any(k in text_lower for k in ["startup", "business", "invest", "market", "revenue"]):
        cat = "business"
    else:
        cat = "other"

    return {
        "category": cat,
        "score": 5,
        "summary": url.split("?")[0].split("/")[-1][:50],
        "tags": [cat.lower()],
    }


# ─── Content Extraction ─────────────────────────────────────────────────────────

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def extract_content(url: str, timeout: int = 30) -> str:
    """
    Extract main content from a URL.
    Tries Jina Reader first, falls back to direct fetch + readability heuristic.
    """
    # Try Jina Reader (best for JS-heavy sites)
    if JINA_API_KEY:
        try:
            resp = requests.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", "").strip()
            if content and len(content) > 200:
                log.debug(f"Jina extracted {len(content)} chars from {url}")
                return content
        except Exception as e:
            log.debug(f"Jina extract failed for {url}: {e}")

    # Fallback: direct fetch with simple readability extraction
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; KnowledgeBot/1.0; +https://example.com/bot)"
        })
        resp.raise_for_status()
        text = resp.text
        content = _simple_readability(text)
        if len(content) > 200:
            return content
    except Exception as e:
        log.warning(f"Direct fetch failed for {url}: {e}")

    return ""


def _simple_readability(html: str) -> str:
    """Strip HTML tags and extract visible text."""
    # Remove script and style sections
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.I)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── X / Twitter ────────────────────────────────────────────────────────────────

X_BEARER = os.getenv("X_BEARER_TOKEN", "")
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")
X_USER_ID = os.getenv("X_USER_ID", "")

PROCESSED_BOOKMARKS_FILE = DATA_DIR / "processed_bookmarks.json"


def get_x_bookmarks() -> list[dict]:
    """Fetch new X (Twitter) bookmarks via API v2."""
    if not X_BEARER:
        log.debug("X_BEARER_TOKEN not set — skipping X bookmarks")
        return []

    processed_ids = set()
    if PROCESSED_BOOKMARKS_FILE.exists():
        try:
            processed_ids = set(json.loads(PROCESSED_BOOKMARKS_FILE.read_text()))
        except Exception:
            pass

    new_bookmarks = []

    try:
        headers = {"Authorization": f"Bearer {X_BEARER}"}
        params = {"max_results": 100, "tweet.fields": "created_at,url,text"}

        resp = requests.get(
            f"https://api.twitter.com/2/users/{X_USER_ID}/bookmarks",
            headers=headers,
            params=params,
            timeout=20,
        )

        if resp.status_code == 401:
            log.error("X token expired or invalid — refresh at https://developer.x.com")
            return []
        resp.raise_for_status()

        tweets = resp.json().get("data", [])
        log.info(f"X: fetched {len(tweets)} bookmarks, {len(processed_ids)} already processed")

        for tweet in tweets:
            tweet_id = tweet["id"]
            if tweet_id in processed_ids:
                continue

            # Extract URL from tweet text or entities
            url = ""
            text = tweet.get("text", "")
            url_match = re.search(r"https?://\S+", text)
            if url_match:
                url = url_match.group(0).rstrip(".,;:)")
            else:
                # No URL in tweet — skip (bookmark without link is just a thought)
                log.debug(f"Skipping bookmark {tweet_id} — no URL")
                processed_ids.add(tweet_id)
                continue

            new_bookmarks.append({
                "id": tweet_id,
                "url": url,
                "text": text,
                "source": "x",
                "bookmarked_at": tweet.get("created_at", ""),
            })
            processed_ids.add(tweet_id)

    except Exception as e:
        log.error(f"Failed to fetch X bookmarks: {e}")

    # Persist processed IDs
    PROCESSED_BOOKMARKS_FILE.write_text(json.dumps(list(processed_ids), ensure_ascii=False))
    return new_bookmarks


# ─── RSS Feeds ─────────────────────────────────────────────────────────────────

FEEDS_CONFIG_FILE = DATA_DIR / "feeds.json"
FEED_STATE_FILE = DATA_DIR / "feed_state.json"


def get_feeds_config() -> dict[str, str]:
    """Load feed URLs from data/feeds.json."""
    if FEEDS_CONFIG_FILE.exists():
        try:
            return json.loads(FEEDS_CONFIG_FILE.read_text())
        except Exception:
            pass
    # Default feeds
    return {
        "Hacker News": "https://hnrss.org/frontpage",
        "AI/ML (ArXiv cs.AI)": "https://arxiv.org/rss/cs.AI",
        "TechCrunch": "https://techcrunch.com/feed/",
    }


def get_new_rss_items() -> list[dict]:
    """Fetch new items from all configured RSS feeds."""
    try:
        import feedparser
    except ImportError:
        log.error("feedparser not installed — run: pip install feedparser")
        return []

    feeds = get_feeds_config()

    state = {}
    if FEED_STATE_FILE.exists():
        try:
            state = json.loads(FEED_STATE_FILE.read_text())
        except Exception:
            pass

    all_items = []
    now = datetime.datetime.utcnow()

    for name, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            last_run = state.get(url, "1970-01-01T00:00:00Z")

            for entry in feed.entries:
                published = _parse_feed_date(entry.get("published", ""))
                if published and str(published) > last_run:
                    all_items.append({
                        "id": entry.get("id", entry.link),
                        "url": entry.link,
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "source": "rss",
                        "feed_name": name,
                        "published": entry.get("published", ""),
                    })

            state[url] = now.isoformat() + "Z"

        except Exception as e:
            log.error(f"Failed to parse feed {name} ({url}): {e}")

    if state:
        FEED_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    log.info(f"RSS: fetched {len(all_items)} new items from {len(feeds)} feeds")
    log.info(f"  Sample items: {[{'title': i['title'][:40], 'published': i['published']} for i in all_items[:3]]}")
    return all_items


def _parse_feed_date(date_str: str) -> Optional[datetime.datetime]:
    """Parse common feed date formats, always returning UTC."""
    if not date_str:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(date_str.strip(), fmt)
            # Normalize to UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                dt = dt.astimezone(datetime.timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ─── Markdown Writer ────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def save_raw_markdown(url: str, content: str, meta: dict, source: str):
    """Write a piece of content to raw/<category>/."""
    category = meta.get("category", "other")
    category = category.lower().replace("_", "-")
    if category not in CATEGORIES:
        category = "other"

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = slugify(meta.get("summary", url.split("/")[-1]))[:50]
    filename = f"{timestamp}_{slug}.md"

    filepath = RAW_DIR / category / filename

    frontmatter = f"""---
source: {url}
category: {category}
score: {meta.get("score", 5)}
date: {datetime.datetime.utcnow().isoformat()}Z
source_type: {source}
tags: {json.dumps(meta.get("tags", []), ensure_ascii=False)}
summary: "{meta.get("summary", "")}"
---

# {meta.get("summary", "Untitled")}

## 原文

{content[:8000]}{"..." if len(content) > 8000 else ""}
"""

    filepath.write_text(frontmatter, encoding="utf-8")
    log.info(f"  ✓ Saved: {filepath.relative_to(REPO_ROOT)} (score={meta.get('score', '?')})")
    return filepath


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Thread-safe token bucket rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 60):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


# ─── Item Processor ────────────────────────────────────────────────────────────

def process_item(item: dict, source: str) -> tuple[bool, str, str]:
    """Process a single item: extract content, classify, save. Returns (success, url, error)."""
    url = item["url"]
    try:
        content = extract_content(url) or item.get("summary", "")
        if not content or len(content) < 100:
            return (False, url, "content too short")
        meta = llm_classify(content, url)
        save_raw_markdown(url, content, meta, source=source)
        return (True, url, "")
    except Exception as e:
        return (False, url, str(e))


# ─── Main ───────────────────────────────────────────────────────────────────────

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
rate_limit = RateLimiter(calls_per_minute=30)  # Be respectful to APIs


def main():
    log.info(f"=== Ingestion run started at {datetime.datetime.utcnow().isoformat()}Z ===")
    log.info(f"Provider: {LLM_PROVIDER}, Max workers: {MAX_WORKERS}")

    total_saved = 0
    errors = 0
    processed_urls = set()

    # ── 1. X Bookmarks ────────────────────────────────────────────────────────
    log.info("─── Fetching X Bookmarks ───")
    x_items = get_x_bookmarks()
    if x_items:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item, "x"): item for item in x_items}
            for future in as_completed(futures):
                rate_limit.wait()
                success, url, err = future.result()
                if success:
                    total_saved += 1
                    processed_urls.add(url)
                else:
                    log.error(f"  ✗ Failed X bookmark {url}: {err}")
                    errors += 1

    # ── 2. RSS Feeds ──────────────────────────────────────────────────────────
    log.info("─── Fetching RSS Feeds ───")
    rss_items = get_new_rss_items()
    log.info(f"Got {len(rss_items)} new RSS items")

    if rss_items:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item, "rss"): item for item in rss_items}
            for future in as_completed(futures):
                rate_limit.wait()
                success, url, err = future.result()
                if success:
                    total_saved += 1
                    processed_urls.add(url)
                else:
                    log.error(f"  ✗ Failed RSS item {url}: {err}")
                    errors += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info(f"=== Ingestion complete: {total_saved} saved, {errors} errors ===")
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

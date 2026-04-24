#!/usr/bin/env python3
"""
Post-processing: read raw/ content and generate wiki/concepts/ and wiki/summaries/.

This script runs locally (not in CI) after you've synced raw/ from the repo.
It uses a stronger LLM (GPT-4o or Claude Sonnet) to:
  1. Identify concepts and create concept pages
  2. Write summaries of high-score items
  3. Update the wiki index

Usage:
  python scripts/update_wiki.py [--score-threshold 7] [--category AI-ML]
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests

# ─── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("update_wiki")

# ─── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.resolve()
RAW_DIR = REPO_ROOT / "raw"
WIKI_DIR = REPO_ROOT / "wiki"
CONCEPTS_DIR = WIKI_DIR / "concepts"
SUMMARIES_DIR = WIKI_DIR / "summaries"
INDEX_FILE = WIKI_DIR / "index.md"

CATEGORIES = ["AI-ML", "coding", "design", "productivity", "research", "business", "other"]

# ─── LLM Setup ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


def llm_generate(system_prompt: str, user_content: str, model: str = "gpt-4o") -> str:
    """Generate content using configured LLM provider."""
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.7,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"No API key available for LLM provider: {LLM_PROVIDER}")


# ─── Concept Extractor ─────────────────────────────────────────────────────────

SYSTEM_CONCEPT = """你是一个知识整理专家。阅读下面的内容，提取核心概念，生成一个概念页面。

要求：
- 用中文写作
- 格式为 Obsidian markdown
- 包含概念定义、关键要点、相关链接
- 链接到其他概念时使用 [[概念名]] wikilink 格式
- 最后附上原文链接

输出格式：
## 概念定义
（2-3句话）

## 关键要点
- ...
- ...

## 相关概念
[[其他概念]]

## 原文
[链接](url)
"""


def extract_concepts(raw_file: Path, content: str, url: str) -> list[dict]:
    """Extract concepts from a raw file and return list of {name, body}."""
    try:
        response = llm_generate(
            SYSTEM_CONCEPT,
            f"URL: {url}\n\n内容：\n{content[:5000]}",
            model="gpt-4o-mini",
        )
        # Parse response to extract concept names (simple heuristic)
        concept_names = set()
        for line in response.split("\n"):
            if line.startswith("##"):
                name = line.lstrip("#").strip()
                concept_names.add(name)
        return [{"name": n, "body": response, "source_url": url}] for n in concept_names]
    except Exception as e:
        log.error(f"Concept extraction failed for {raw_file}: {e}")
        return []


# ─── Summary Generator ──────────────────────────────────────────────────────────

SYSTEM_SUMMARY = """你是一个文风润色专家。阅读下面的内容，用流畅的中文改写，保留核心信息。

要求：
- 用中文写作
- 保留关键数据和引用
- 长度适中（300-800字）
- 结构清晰（小标题分段）
- 链接回原文
"""


def generate_summary(raw_file: Path, content: str, meta: dict) -> str:
    """Generate a polished summary from raw content."""
    try:
        summary = llm_generate(
            SYSTEM_SUMMARY,
            f"原文链接：{meta['url']}\n\n分类：{meta['category']}\n\n内容：\n{content[:6000]}",
            model="gpt-4o-mini",
        )
        frontmatter = f"""---
title: "{meta.get('summary', 'Untitled')}"
source: {meta['url']}
date: {datetime.datetime.utcnow().date().isoformat()}
category: {meta['category']}
score: {meta.get('score', 5)}
tags: {json.dumps(meta.get('tags', []), ensure_ascii=False)}
---

"""
        return frontmatter + "# " + (meta.get("summary", "Untitled")) + "\n\n" + summary
    except Exception as e:
        log.error(f"Summary generation failed for {raw_file}: {e}")
        return ""


# ─── Index Updater ─────────────────────────────────────────────────────────────

def update_index():
    """Regenerate wiki/index.md from all concept and summary files."""
    concepts = sorted([f.stem for f in CONCEPTS_DIR.glob("*.md")])
    summaries = sorted([f.stem for f in SUMMARIES_DIR.glob("*.md")])

    index = f"""# 知识库索引

> 自动生成，最后更新：{datetime.datetime.utcnow().isoformat()}Z

## 概念 ({len(concepts)})

{" ".join(f"[[{c}]]" for c in concepts)}

## 摘要 ({len(summaries)})

{" ".join(f"[[{s}]]" for s in summaries)}

---

*此文件由 scripts/update_wiki.py 自动生成*
"""
    INDEX_FILE.write_text(index, encoding="utf-8")
    log.info(f"Updated index: {len(concepts)} concepts, {len(summaries)} summaries")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Update wiki from raw content")
    parser.add_argument("--score-threshold", type=int, default=7, help="Min score to generate summary")
    parser.add_argument("--category", default=None, help="Process only specific category")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot run update_wiki.py")
        sys.exit(1)

    log.info("=== Wiki update started ===")

    for category in CATEGORIES:
        if args.category and args.category != category:
            continue
        cat_dir = RAW_DIR / category
        if not cat_dir.exists():
            continue

        for raw_file in cat_dir.glob("*.md"):
            try:
                text = raw_file.read_text(encoding="utf-8")
                # Parse frontmatter
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1]
                        body = parts[2]
                        meta = dict(re.findall(r"(\w+): (.+)", frontmatter))
                        meta["url"] = re.search(r"source: (.+)", frontmatter)
                        if meta["url"]:
                            meta["url"] = meta["url"].group(1)
                        score = int(meta.get("score", 5))
                    else:
                        body = text
                        meta = {"url": "", "category": category, "score": 5, "summary": raw_file.stem}
                else:
                    body = text
                    meta = {"url": "", "category": category, "score": 5, "summary": raw_file.stem}

                url = meta.get("url", "")

                # Generate summary if score is high enough
                if score >= args.score_threshold:
                    summary_file = SUMMARIES_DIR / f"{raw_file.stem}.md"
                    if args.dry_run:
                        log.info(f"  [DRY RUN] Would generate summary: {raw_file.stem}")
                    else:
                        summary = generate_summary(raw_file, body, meta)
                        if summary:
                            summary_file.write_text(summary, encoding="utf-8")
                            log.info(f"  ✓ Summary: {summary_file.name}")

                # Extract and save concepts (every file)
                if not args.dry_run:
                    concepts = extract_concepts(raw_file, body, url)
                    for concept in concepts:
                        safe_name = re.sub(r"[^\w\s-]", "", concept["name"]).strip()
                        safe_name = re.sub(r"[-\s]+", "-", safe_name).lower()
                        concept_file = CONCEPTS_DIR / f"{safe_name}.md"
                        concept_file.write_text(concept["body"], encoding="utf-8")
                        log.info(f"  ✓ Concept: {concept_file.name}")

            except Exception as e:
                log.error(f"  ✗ Failed to process {raw_file}: {e}")

    update_index()
    log.info("=== Wiki update complete ===")


if __name__ == "__main__":
    main()

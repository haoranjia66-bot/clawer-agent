# coding=utf-8
"""
Hugging Face Daily Papers 解析器（非 RSS）

Hugging Face 的 Daily Papers 页面不是 RSS/Atom/JSON Feed。
本模块将页面内容解析为 ParsedRSSItem 列表，以复用现有 RSS 入库/去重/增量/推送链路。
"""

from __future__ import annotations

import html as _html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .parser import ParsedRSSItem


_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{5})\b")
_HF_PAPERS_PATH_RE = re.compile(r"/papers/(\d{4}\.\d{5})\b")
_HF_DATE_RE = re.compile(r"/papers/date/(\d{4}-\d{2}-\d{2})\b")


def is_hf_daily_papers_url(url: str) -> bool:
    if not url:
        return False
    # 覆盖 today 页 + date 页
    return url.startswith("https://huggingface.co/papers")


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _published_at_for_feed(feed_url: str) -> str:
    """
    - date 页：固定到当天中午 UTC（降低跨时区边界造成的“天数差”误差）
    - today 页：使用抓取时刻 UTC
    """
    m = _HF_DATE_RE.search(feed_url or "")
    if not m:
        return _now_iso_utc()
    return f"{m.group(1)}T12:00:00Z"


def _strip_tags(s: str) -> str:
    # 轻量级去标签：用于 fallback（不引入 bs4/lxml）
    s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\\1>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _walk_json(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_json(it)


def _extract_candidates_from_next_data(next_data: Dict[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
    """
    从 Next.js 的 __NEXT_DATA__ 中尽量抽取 (arxiv_id, title, author_text)。
    该结构是内部实现，字段名可能变化，因此采用“宽松匹配 + 去重”策略。
    """
    out: List[Tuple[str, str, Optional[str]]] = []

    for d in _walk_json(next_data):
        title = d.get("title") if isinstance(d.get("title"), str) else None
        if not title or len(title.strip()) < 6:
            continue

        # arXiv id 可能出现在多个字段
        arxiv_id = None
        for k in ("arxivId", "arxiv_id", "paperId", "paper_id", "id", "paper_id"):
            v = d.get(k)
            if isinstance(v, str):
                m = _ARXIV_ID_RE.search(v)
                if m:
                    arxiv_id = m.group(1)
                    break

        # 有些结构会把 URL 放在 url 字段里
        if not arxiv_id:
            url = d.get("url") if isinstance(d.get("url"), str) else None
            if url:
                m = _ARXIV_ID_RE.search(url)
                if m:
                    arxiv_id = m.group(1)

        if not arxiv_id:
            continue

        # 作者信息字段（不保证存在）
        author = None
        for ak in ("authors", "author", "authorText", "author_text"):
            av = d.get(ak)
            if isinstance(av, str) and av.strip():
                author = av.strip()
                break

        out.append((arxiv_id, title.strip(), author))

    # 去重：同一个 arxiv_id 取最长 title（更像正式标题）
    dedup: Dict[str, Tuple[str, Optional[str]]] = {}
    for arxiv_id, title, author in out:
        if arxiv_id not in dedup or len(title) > len(dedup[arxiv_id][0]):
            dedup[arxiv_id] = (title, author)

    return [(k, v[0], v[1]) for k, v in dedup.items()]


def _try_parse_next_data(html: str) -> List[Tuple[str, str, Optional[str]]]:
    m = re.search(
        r'(?is)<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html or "",
    )
    if not m:
        return []

    raw = m.group(1).strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    return _extract_candidates_from_next_data(data if isinstance(data, dict) else {})


def _parse_from_html_links(html: str) -> List[Tuple[str, str]]:
    """
    Fallback：从 HTML 中尽量提取 <a href="/papers/<id>">Title</a> 的 title。
    同时规避“upvote 数字链接”这类只包含数字的 anchor。
    """
    candidates: List[Tuple[str, str]] = []

    # 先尝试更“紧”的模式
    for m in re.finditer(
        r'(?is)href="/papers/(?P<id>\d{4}\.\d{5})(?:[^"]*)"\s*[^>]*>(?P<title>[^<]{3,300})</a>',
        html or "",
    ):
        arxiv_id = m.group("id")
        title = _html.unescape(m.group("title"))
        title = re.sub(r"\s+", " ", title).strip()
        if not title or re.fullmatch(r"[\d\.\-kK]+", title):
            continue
        candidates.append((arxiv_id, title))

    # 去重：同一个 id 取最长 title
    dedup: Dict[str, str] = {}
    for arxiv_id, title in candidates:
        if arxiv_id not in dedup or len(title) > len(dedup[arxiv_id]):
            dedup[arxiv_id] = title

    return list(dedup.items())


def _parse_from_markdownish_text(text: str) -> List[Tuple[str, str]]:
    """
    极端兜底：如果返回内容被代理/缓存改写成“类 Markdown”的文本（例如某些抓取器），
    解析形如：
      [333](.../papers/2602.05400)
      ### Title...
    """
    candidates: List[Tuple[str, str]] = []
    for m in re.finditer(
        r"(?is)\(/papers/(?P<id>\d{4}\.\d{5})\)[^\n]*\n\s*###\s+(?P<title>.+?)\s*(?:\n|$)",
        text or "",
    ):
        arxiv_id = m.group("id")
        title = m.group("title").strip()
        title = re.sub(r"\s+", " ", title)
        if title:
            candidates.append((arxiv_id, title))

    # 再兜底一层：只拿到 ID 列表时也返回（title 用 paper id 占位，避免完全空）
    if not candidates:
        ids = sorted(set(_HF_PAPERS_PATH_RE.findall(text or "")))
        for arxiv_id in ids:
            candidates.append((arxiv_id, f"arXiv:{arxiv_id}"))

    dedup: Dict[str, str] = {}
    for arxiv_id, title in candidates:
        if arxiv_id not in dedup or len(title) > len(dedup[arxiv_id]):
            dedup[arxiv_id] = title
    return list(dedup.items())


def parse_hf_daily_papers(content: str, feed_url: str) -> List[ParsedRSSItem]:
    """
    将 Hugging Face Daily Papers 页面内容解析为 ParsedRSSItem 列表。
    """
    published_at = _published_at_for_feed(feed_url)

    # 1) 优先：__NEXT_DATA__（更稳定）
    next_candidates = _try_parse_next_data(content)
    if next_candidates:
        items: List[ParsedRSSItem] = []
        for arxiv_id, title, author in next_candidates:
            items.append(
                ParsedRSSItem(
                    title=title,
                    url=f"https://huggingface.co/papers/{arxiv_id}",
                    published_at=published_at,
                    summary=None,
                    author=author,
                    guid=f"hf-papers:{arxiv_id}",
                )
            )
        return items

    # 2) HTML anchor fallback
    link_candidates = _parse_from_html_links(content)
    if link_candidates:
        return [
            ParsedRSSItem(
                title=title,
                url=f"https://huggingface.co/papers/{arxiv_id}",
                published_at=published_at,
                summary=None,
                author=None,
                guid=f"hf-papers:{arxiv_id}",
            )
            for arxiv_id, title in link_candidates
        ]

    # 3) markdown-ish text fallback
    md_candidates = _parse_from_markdownish_text(content)
    if md_candidates:
        return [
            ParsedRSSItem(
                title=title,
                url=f"https://huggingface.co/papers/{arxiv_id}",
                published_at=published_at,
                summary=None,
                author=None,
                guid=f"hf-papers:{arxiv_id}",
            )
            for arxiv_id, title in md_candidates
        ]

    # 4) 最后：完全去标签后扫 arXiv id（几乎只用于 debug）
    text = _strip_tags(content or "")
    ids = sorted(set(_ARXIV_ID_RE.findall(text)))
    return [
        ParsedRSSItem(
            title=f"arXiv:{arxiv_id}",
            url=f"https://huggingface.co/papers/{arxiv_id}",
            published_at=published_at,
            summary=None,
            author=None,
            guid=f"hf-papers:{arxiv_id}",
        )
        for arxiv_id in ids
    ]


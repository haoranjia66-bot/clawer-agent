# coding=utf-8
"""
论文/技术文章短摘要（100字）生成器

用于 RSS 论文源（如 arXiv）在推送中为每条条目生成固定长度的中文 summary。
优先使用 AI（若已配置），否则降级为从原 abstract/title 截断。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from trendradar.ai.client import AIClient

BATCH_SIZE = 10
FALLBACK_MAX_CHARS = 200


def _strip_ws(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate_chars(text: str, max_chars: int) -> str:
    text = _strip_ws(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """截断到 max_chars 以内，尽量在句号/逗号/分号处断开，避免单词截断。"""
    text = _strip_ws(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    chunk = text[:max_chars]
    for sep in [". ", "。", "；", "; ", ", ", "，"]:
        pos = chunk.rfind(sep)
        if pos > max_chars // 3:
            return chunk[: pos + len(sep)].rstrip()

    space_pos = chunk.rfind(" ")
    if space_pos > max_chars // 3:
        return chunk[:space_pos].rstrip() + "..."
    return chunk.rstrip() + "..."


def _extract_json(text: str) -> str:
    """从模型回复中提取 JSON（兼容 ```json ...``` 或 ``` ...```）。"""
    if not text:
        return ""
    raw = text.strip()
    if "```json" in raw:
        part = raw.split("```json", 1)[1]
        end = part.find("```")
        return (part[:end] if end != -1 else part).strip()
    if "```" in raw:
        parts = raw.split("```", 2)
        if len(parts) >= 2:
            return parts[1].strip()
    return raw


@dataclass
class PaperSummaryRequest:
    key: str
    title: str
    abstract: str


class PaperSummarizer:
    def __init__(self, ai_config: Dict[str, Any], max_chars: int = 100):
        self.ai_config = ai_config or {}
        self.max_chars = max_chars
        self.client = AIClient(self.ai_config)

    def _ai_enabled(self) -> bool:
        ok, _ = self.client.validate_config()
        return ok

    def _fallback_summary(self, req: PaperSummaryRequest) -> str:
        """AI 不可用时，从 abstract 或 title 截断生成 fallback 摘要。"""
        text = _strip_ws(req.abstract) or _strip_ws(req.title)
        if not text:
            return ""
        return _truncate_at_sentence(text, FALLBACK_MAX_CHARS)

    def _call_ai_batch(self, items: List[Dict]) -> Dict[str, str]:
        """对一批 items 调用 AI，返回 {key: summary}。"""
        system = (
            "你是一名中文论文摘要助手。基于论文标题与摘要，生成极短中文总结。"
            "输出必须严格为 JSON，不要输出任何额外文字。"
        )
        user = (
            "请为每条输入生成一段【不超过100个汉字】的简体中文摘要，要求：\n"
            "- 只输出一句话或两句短句，但不得换行（最终值中不得包含\\n）\n"
            "- 聚焦：研究问题/方法/贡献/结果（尽量覆盖）\n"
            "- 不要写\u201c本文提出/本文研究\u201d这类套话\n"
            "- 尽量不要出现英文；若不可避免，保留必要缩写即可\n"
            "- 不要编号，不要Markdown\n"
            "- 若信息不足，用标题信息做合理概括\n"
            "\n"
            "输入 JSON 数组：\n"
            f"{json.dumps(items, ensure_ascii=False)}\n"
            "\n"
            "输出 JSON 对象，key 为输入的 key，value 为摘要字符串：\n"
            '{"<key>":"<summary>"}'
        )

        max_tok = max(len(items) * 120, 800)

        resp = self.client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=max_tok,
        )

        json_str = _extract_json(resp)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return {}
        return data

    def summarize_batch(self, reqs: List[PaperSummaryRequest]) -> Dict[str, str]:
        """
        返回 {key: short_summary}。
        AI 可用时分批调用 AI；不可用或调用失败时，降级为 abstract/title 截断。
        fallback 结果会标记 _is_fallback=True 供调用方判断是否写入缓存。
        """
        if not reqs:
            return {}

        use_ai = self._ai_enabled()
        if not use_ai:
            print("[RSS] AI 未配置，使用 abstract/title 截断作为短摘要（不写入缓存）")
            out = {}
            for r in reqs:
                fb = self._fallback_summary(r)
                if fb:
                    out[r.key] = fb
            self._last_is_fallback = True
            return out

        self._last_is_fallback = False

        prep = {}
        for r in reqs:
            title = _strip_ws(r.title)
            abstract = _truncate_chars(r.abstract, 1200)
            prep[r.key] = {"key": r.key, "title": title, "abstract": abstract}

        keys = list(prep.keys())
        out: Dict[str, str] = {}

        for i in range(0, len(keys), BATCH_SIZE):
            batch_keys = keys[i : i + BATCH_SIZE]
            batch_items = [prep[k] for k in batch_keys]
            batch_reqs = {r.key: r for r in reqs if r.key in batch_keys}

            try:
                ai_result = self._call_ai_batch(batch_items)
            except Exception as e:
                print(f"[RSS] AI 批次调用失败（第 {i // BATCH_SIZE + 1} 批），降级为截断: {e}")
                for k in batch_keys:
                    r = batch_reqs.get(k)
                    if r:
                        out[k] = self._fallback_summary(r)
                self._last_is_fallback = True
                continue

            for k in batch_keys:
                v = ai_result.get(k, "")
                v = _strip_ws(str(v))
                v = v.replace("\n", " ").replace("\r", " ")
                if v:
                    out[k] = _truncate_chars(v, self.max_chars)
                else:
                    r = batch_reqs.get(k)
                    if r:
                        out[k] = self._fallback_summary(r)
                        self._last_is_fallback = True

        return out

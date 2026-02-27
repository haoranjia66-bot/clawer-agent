# coding=utf-8
"""
论文/技术文章短摘要（100字）生成器

用于 RSS 论文源（如 arXiv）在推送中为每条条目生成固定长度的中文 summary。
优先使用 AI（若已配置），否则降级为从原 summary 截断。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from trendradar.ai.client import AIClient


def _strip_ws(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate_chars(text: str, max_chars: int) -> str:
    """
    以“字符数”截断（近似满足“100字”需求）。
    """
    text = _strip_ws(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _extract_json(text: str) -> str:
    """
    从模型回复中提取 JSON（兼容 ```json ...``` 或 ``` ...```）。
    """
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

    def summarize_batch(self, reqs: List[PaperSummaryRequest]) -> Dict[str, str]:
        """
        返回 {key: short_summary}
        """
        if not reqs:
            return {}

        # 无 AI 配置：不生成（避免把英文 abstract 缓存成“短摘要”）
        if not self._ai_enabled():
            return {}

        # 构建输入 payload，控制体积：abstract 过长先截断
        items = []
        for r in reqs:
            title = _strip_ws(r.title)
            abstract = _truncate_chars(r.abstract, 1200)  # 避免单条太长
            items.append({"key": r.key, "title": title, "abstract": abstract})

        system = (
            "你是一名中文论文摘要助手。基于论文标题与摘要，生成极短中文总结。"
            "输出必须严格为 JSON，不要输出任何额外文字。"
        )
        user = (
            "请为每条输入生成一段【不超过100个汉字】的简体中文摘要，要求：\n"
            "- 只输出一句话或两句短句，但不得换行（最终值中不得包含\\n）\n"
            "- 聚焦：研究问题/方法/贡献/结果（尽量覆盖）\n"
            "- 不要写“本文提出/本文研究”这类套话\n"
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

        resp = self.client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=800,
        )

        json_str = _extract_json(resp)
        try:
            data = json.loads(json_str)
        except Exception:
            # 解析失败：不写入（避免缓存脏数据）
            return {}

        out: Dict[str, str] = {}
        if isinstance(data, dict):
            for r in reqs:
                v = data.get(r.key, "")
                v = _strip_ws(str(v))
                v = v.replace("\n", " ").replace("\r", " ")
                out[r.key] = _truncate_chars(v, self.max_chars) if v else ""
        else:
            return {}

        return out


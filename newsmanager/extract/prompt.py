"""Prompt template, output schema, and input truncation.

The prompt and JSON schema are plain files under prompts/ and are meant to be
edited. Both are content-hashed into a `prompt_version`, which is stamped on
every extracted row -- so results produced by different prompt revisions never
silently mix, and a prompt change is a re-run you can scope rather than a
corruption you discover later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDERS = ("{{ARTICLE}}", "{{TITLE}}", "{{DATE}}", "{{SYMBOLS}}")

# Truncation default, measured on 19,320 articles sampled across the corpus:
#
#   article chars: p50=3,518  p75=5,326  p90=6,976  p99=32,285  max=272,968
#
# The tail is extreme -- the longest article is ~68k tokens and would alone cost
# more than a hundred median ones. Capping at 8,000 chars keeps 85.9% of all
# tokens while truncating only 6.3% of articles; 12,000 keeps 89.2% at 2.2%
# truncated. Past that the curve is flat and you are paying for outliers.
DEFAULT_MAX_CHARS = 8000


@dataclass(frozen=True)
class Prompt:
    template: str
    schema: dict
    version: str
    max_chars: int

    def render(self, *, article: str, title: str | None, date: str | None, symbols: str | None) -> tuple[str, bool]:
        """Fill the template. Returns (prompt_text, was_truncated)."""
        body = article or ""
        truncated = len(body) > self.max_chars
        if truncated:
            # Cut at a whitespace boundary so the model never sees a half word,
            # and say so explicitly rather than letting the text just stop.
            cut = body[: self.max_chars]
            sp = cut.rfind(" ")
            if sp > self.max_chars * 0.9:
                cut = cut[:sp]
            body = cut + "\n\n[article truncated]"
        out = self.template
        for key, val in (
            ("{{ARTICLE}}", body),
            ("{{TITLE}}", title or "(untitled)"),
            ("{{DATE}}", date or "(unknown)"),
            ("{{SYMBOLS}}", symbols or "(none)"),
        ):
            out = out.replace(key, val)
        return out, truncated


def load(prompt_path: Path, schema_path: Path, max_chars: int = DEFAULT_MAX_CHARS) -> Prompt:
    template = Path(prompt_path).read_text(encoding="utf-8")
    if "{{ARTICLE}}" not in template:
        raise ValueError(f"{prompt_path} must contain the {{{{ARTICLE}}}} placeholder")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    # Version covers prompt, schema and truncation together: all three change
    # what the model actually saw, so all three belong in the identity.
    h = hashlib.sha256()
    h.update(template.encode())
    h.update(json.dumps(schema, sort_keys=True).encode())
    h.update(str(max_chars).encode())
    return Prompt(template=template, schema=schema, version=h.hexdigest()[:12], max_chars=max_chars)


def estimate_tokens(chars: int) -> int:
    """Rough English-text token count. Used for planning only, never for billing."""
    return max(1, chars // 4)

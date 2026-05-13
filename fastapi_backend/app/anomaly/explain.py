"""LLM-backed explanations for flagged anomalies.

Two entry points:

* `explain_anomalies(anomalies)` — file-level summary, called from
  `/logs/files/{id}/summary` so the SOC analyst sees a paragraph above
  the per-row table. Returns None if no API key is configured.

* `explain_entry(entry)` — per-entry verdict + description, called from
  `/logs/entries/{id}/explain` when the analyst clicks "Explain" on a
  single row. Returns a dict shaped:
      {"verdict": "anomaly" | "false_positive", "description": "..."}

Provider is auto-selected from environment:
  ANTHROPIC_API_KEY -> Claude (preferred)
  GEMINI_API_KEY    -> Gemini (fallback)
  neither           -> functions return None / a stub dict

Implementation uses httpx directly (already a FastAPI dependency) so we
don't need to pin the Anthropic / Google SDKs in the runtime image.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

LOG = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
# Cheapest *currently-served* Haiku. The older claude-3-haiku-20240307
# was retired and now 404s. Haiku 4.5 is roughly $1/$5 per MTok — still
# negligible for a single-entry verdict that returns a short JSON object.
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

MAX_ANOMALIES_FOR_PROMPT = 25
REQUEST_TIMEOUT = 20.0  # seconds


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def explain_anomalies(anomalies: list[dict]) -> Optional[str]:
    """File-level summary. Returns None if no API key configured."""
    if not anomalies:
        return None
    prompt = _build_file_prompt(anomalies)
    return _call_llm(prompt, max_tokens=400)


def explain_entry(entry: dict) -> dict:
    """Per-entry verdict + description.

    Always returns a dict so the frontend has a consistent shape. If no
    LLM is configured we fall back to echoing the rule-based reason.
    """
    if not _has_api_key():
        return {
            "verdict": "anomaly" if entry.get("is_anomaly") else "normal",
            "description": (
                entry.get("anomaly_reason")
                or "No LLM API key configured; showing the model's reason only."
            ),
            "source": "fallback",
        }

    prompt = _build_entry_prompt(entry)
    raw = _call_llm(prompt, max_tokens=300)
    if raw is None:
        return {
            "verdict": "anomaly" if entry.get("is_anomaly") else "normal",
            "description": "LLM call failed; see backend logs.",
            "source": "fallback",
        }
    parsed = _parse_entry_response(raw)
    parsed["source"] = "llm"
    return parsed


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _serialize_entry(a: dict) -> dict:
    return {
        "timestamp": str(a.get("timestamp")) if a.get("timestamp") else None,
        "source_ip": a.get("source_ip"),
        "user": a.get("user_login"),
        "method": a.get("method"),
        "url": a.get("url"),
        "status_code": a.get("status_code"),
        "action": a.get("action"),
        "threat_name": a.get("threat_name"),
        "url_category": a.get("url_category"),
        "anomaly_score": a.get("anomaly_score"),
        "anomaly_reason": a.get("anomaly_reason"),
    }


def _build_file_prompt(anomalies: list[dict]) -> str:
    sample = [_serialize_entry(a) for a in anomalies[:MAX_ANOMALIES_FOR_PROMPT]]
    return (
        "You are assisting a SOC analyst. Below is a list of anomalous web "
        "proxy log entries flagged by an automated detector. Produce a short "
        "(<=150 word) summary that a human can act on: name the most "
        "important pattern(s), call out any single entry that looks "
        "high-severity, and suggest one next investigation step.\n\n"
        f"ANOMALIES (JSON):\n{json.dumps(sample, indent=2)}"
    )


def _build_entry_prompt(entry: dict) -> str:
    payload = _serialize_entry(entry)
    return (
        "You are assisting a SOC analyst reviewing ONE flagged ZScaler web "
        "proxy log entry. Decide whether the entry actually looks malicious, "
        "or whether it's a likely false positive from the automated detector.\n\n"
        "Obvious-malicious signals (verdict = anomaly):\n"
        "  - URL contains SQL injection patterns ('--', UNION, ' OR 1=1, "
        "encoded equivalents like %27, waitfor delay)\n"
        "  - URL contains XSS (<script>, javascript:, on*= handlers)\n"
        "  - Path traversal (../, %2e%2e)\n"
        "  - Server-side include / command injection (#exec, ;cat /etc/, "
        "<!--#include)\n"
        "  - Reconnaissance: requests for .bak/.old/.inc/.java backup files, "
        "admin endpoints, info-disclosure paths\n"
        "  - Tampered parameters (extra '%2F', appended quotes, "
        "field-rename attacks like modoA= instead of modo=)\n\n"
        "Likely-false-positive signals (verdict = false_positive):\n"
        "  - Plain GET to a known page (index.jsp, productos.jsp, an image)\n"
        "  - Routine POST with sensible Content-Length and no payload "
        "showing in the URL\n"
        "  - Nothing in the URL or parameters looks tampered\n\n"
        "Respond with STRICT JSON, no prose around it:\n"
        '  {"verdict": "anomaly" | "false_positive", '
        '"description": "1-2 sentences for a SOC analyst"}\n\n'
        f"ENTRY (JSON):\n{json.dumps(payload, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _call_llm(prompt: str, *, max_tokens: int = 400) -> Optional[str]:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if anthropic_key:
        return _call_anthropic(prompt, anthropic_key, max_tokens)
    if gemini_key:
        return _call_gemini(prompt, gemini_key, max_tokens)
    return None


def _call_anthropic(prompt: str, api_key: str, max_tokens: int) -> Optional[str]:
    try:
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text")
        return None
    except Exception as exc:
        LOG.warning("Anthropic call failed: %s", exc)
        return None


def _call_gemini(prompt: str, api_key: str, max_tokens: int) -> Optional[str]:
    try:
        resp = httpx.post(
            f"{GEMINI_URL}?key={api_key}",
            headers={"content-type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        return parts[0].get("text")
    except Exception as exc:
        LOG.warning("Gemini call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_entry_response(raw: str) -> dict[str, Any]:
    """Coerce the model's response into {verdict, description}.

    Models sometimes wrap JSON in code fences or add a sentence of preamble
    despite the instruction. Strip both and try to recover.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            verdict = str(data.get("verdict", "")).lower().strip()
            description = str(data.get("description", "")).strip()
            if verdict not in ("anomaly", "false_positive"):
                verdict = "anomaly"
            if not description:
                description = raw.strip()
            return {"verdict": verdict, "description": description}
        except json.JSONDecodeError:
            pass

    return {"verdict": "anomaly", "description": raw.strip()}

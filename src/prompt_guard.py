"""
Layer 1 — Prompt Guard

Detect prompt injection risk from raw prompt / action context and
produce a single binary signal used by the downstream feature pipeline.
"""

import re
from typing import Dict, Optional

SUSPICIOUS_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?previous instructions",
    r"ignore (all )?prior instructions",
    r"disregard (all )?prior instructions",
    r"ignore (all )?system instructions",
    r"ignore (all )?safety rules",
    r"bypass (the )?(rules|filters|safety)",
    r"override (the )?(rules|instructions|policy)",
    r"continue (even )?if",
    r"execute.*command",
    r"open the (file|url)",
    r"write a script",
    r"run code",
    r"send .*to .*external",
    r"follow.*instructions.*below",
    r"you are not allowed",
    r"you must",
    r"act as if",
    r"jailbreak",
    r"break out",
    r"self-harm",
    r"poison the",
    r"malware",
    r"exploit",
    r"secret key",
    r"api key",
    r"authentication token",
    r"sensitive information",
    r"private data",
    r"data exfiltration",
]

MIN_SCORE_THRESHOLD = 1


def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_prompt_injection(prompt_text: Optional[str], context: Optional[Dict] = None) -> int:
    """Return 1 if the prompt text or context appears to contain injection signals."""
    normalized = _normalize_text(prompt_text)
    if not normalized:
        return 0

    score = 0
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, normalized):
            score += 1

    if context:
        if context.get("tool_requested") == "external_webhook":
            score += int("send" in normalized or "post" in normalized)
        if context.get("requested_action") in {"delete_record", "export_report", "change_permission"}:
            score += int("bypass" in normalized or "ignore" in normalized)
        if context.get("resource_type") in {"api_key_secret", "system_config"}:
            score += int("secret" in normalized or "token" in normalized)

    return int(score >= MIN_SCORE_THRESHOLD)


def prompt_guard(raw_prompt: str, action_context: Optional[Dict] = None) -> Dict[str, int]:
    """Return the prompt guard signal for a raw prompt/action pair."""
    return {
        "prompt_injection_detected": detect_prompt_injection(raw_prompt, action_context)
    }

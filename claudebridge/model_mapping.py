"""OpenRouter slug to Claude Code model mapping."""

import re

# Simple names that map directly to Claude Code model identifiers
SIMPLE_NAMES: set[str] = {"opus", "sonnet", "haiku"}

# Word-boundary pattern for matching model names in slugs
# Matches model names surrounded by non-alphanumeric chars (or string boundaries)
# e.g. "anthropic/claude-opus-4.5" matches "opus", but "opus-lite" would need "opus" as a segment
_MODEL_PATTERN = re.compile(
    r'(?:^|[^a-zA-Z])(' + '|'.join(sorted(SIMPLE_NAMES)) + r')(?:[^a-zA-Z]|$)',
    re.IGNORECASE,
)

# Available models for /api/v1/models endpoint (OpenRouter-style)
AVAILABLE_MODELS: list[dict[str, str]] = [
    {"slug": f"anthropic/claude-{name}", "name": f"Claude {name.capitalize()}"}
    for name in sorted(SIMPLE_NAMES)
]


class UnsupportedModelError(ValueError):
    """Raised when an unsupported model identifier is provided."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"Unsupported model: '{model}'. "
            f"Supported models: {', '.join(sorted(SIMPLE_NAMES))}, "
            f"or any slug containing 'opus', 'sonnet', or 'haiku'"
        )


def resolve_model(model: str) -> str:
    """Resolve an OpenRouter-style slug or simple name to a Claude Code model.

    Uses word-boundary matching to prevent false positives. Model names must appear
    as distinct segments separated by non-alpha characters (/, -, _, ., etc.).

    Args:
        model: Model identifier (OpenRouter slug or simple name)

    Returns:
        Claude Code model identifier (opus, sonnet, haiku)

    Raises:
        UnsupportedModelError: If model is not recognized
    """
    model_stripped = model.strip()
    model_lower = model_stripped.lower()

    # Already a simple Claude Code name (exact match)
    if model_lower in SIMPLE_NAMES:
        return model_lower

    # Word-boundary match: find model name as a distinct segment in the slug
    match = _MODEL_PATTERN.search(model_lower)
    if match:
        return match.group(1).lower()

    # Unknown model - raise error
    raise UnsupportedModelError(model)

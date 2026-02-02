"""OpenRouter slug to Claude Code model mapping."""

# Mapping from OpenRouter-style slugs to Claude Code model identifiers
# Claude Code uses simple names: opus, sonnet, haiku
OPENROUTER_TO_CLAUDE: dict[str, str] = {
    # Claude 4.5 series
    "anthropic/claude-opus-4.5": "opus",
    "anthropic/claude-sonnet-4.5": "sonnet",
    "anthropic/claude-haiku-4.5": "haiku",
    # Claude 4.1 series
    "anthropic/claude-opus-4.1": "opus",
    # Claude 4 series
    "anthropic/claude-opus-4": "opus",
    "anthropic/claude-sonnet-4": "sonnet",
    # Claude 3.7 series
    "anthropic/claude-3.7-sonnet": "sonnet",
    # Claude 3.5 series
    "anthropic/claude-3.5-haiku": "haiku",
    "anthropic/claude-3.5-haiku-20241022": "haiku",
    "anthropic/claude-3.5-sonnet": "sonnet",
    "anthropic/claude-3.5-sonnet-20240620": "sonnet",
    # Claude 3 series
    "anthropic/claude-3-haiku": "haiku",
    "anthropic/claude-3-sonnet": "sonnet",
    "anthropic/claude-3-opus": "opus",
}

# Simple names that map directly
SIMPLE_NAMES: set[str] = {"opus", "sonnet", "haiku"}

# Available models for /api/v1/models endpoint (OpenRouter-style)
AVAILABLE_MODELS: list[dict[str, str]] = [
    {"slug": "anthropic/claude-opus-4.5", "name": "Claude Opus 4.5"},
    {"slug": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5"},
    {"slug": "anthropic/claude-haiku-4.5", "name": "Claude Haiku 4.5"},
    {"slug": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4"},
    {"slug": "anthropic/claude-opus-4", "name": "Claude Opus 4"},
]


class UnsupportedModelError(ValueError):
    """Raised when an unsupported model identifier is provided."""

    def __init__(self, model: str):
        self.model = model
        supported = list(SIMPLE_NAMES) + list(OPENROUTER_TO_CLAUDE.keys())
        super().__init__(
            f"Unsupported model: '{model}'. "
            f"Supported models: {', '.join(sorted(SIMPLE_NAMES))}, "
            f"or OpenRouter slugs like 'anthropic/claude-sonnet-4'"
        )


def resolve_model(model: str) -> str:
    """Resolve an OpenRouter-style slug or simple name to a Claude Code model.

    Args:
        model: Model identifier (OpenRouter slug or simple name)

    Returns:
        Claude Code model identifier (opus, sonnet, haiku)

    Raises:
        UnsupportedModelError: If model is not recognized
    """
    # Already a simple Claude Code name
    if model.lower() in SIMPLE_NAMES:
        return model.lower()

    # OpenRouter slug lookup
    if model in OPENROUTER_TO_CLAUDE:
        return OPENROUTER_TO_CLAUDE[model]

    # Try case-insensitive lookup
    model_lower = model.lower()
    for slug, claude_model in OPENROUTER_TO_CLAUDE.items():
        if slug.lower() == model_lower:
            return claude_model

    # Unknown model - raise error
    raise UnsupportedModelError(model)

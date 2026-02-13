"""OpenRouter slug to Claude Code model mapping."""

# Simple names that map directly to Claude Code model identifiers
SIMPLE_NAMES: set[str] = {"opus", "sonnet", "haiku"}

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

    Args:
        model: Model identifier (OpenRouter slug or simple name)

    Returns:
        Claude Code model identifier (opus, sonnet, haiku)

    Raises:
        UnsupportedModelError: If model is not recognized
    """
    model_lower = model.lower()

    # Already a simple Claude Code name
    if model_lower in SIMPLE_NAMES:
        return model_lower

    # Match by substring: any slug containing opus, sonnet, or haiku
    for name in SIMPLE_NAMES:
        if name in model_lower:
            return name

    # Unknown model - raise error
    raise UnsupportedModelError(model)

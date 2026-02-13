"""
Unit tests for model mapping functionality.

These tests verify OpenRouter slug resolution and error handling.

Usage:
- pytest tests/test_model_mapping.py -v
"""

import pytest

from claudebridge.model_mapping import (
    resolve_model,
    UnsupportedModelError,
    SIMPLE_NAMES,
    AVAILABLE_MODELS,
)


@pytest.mark.unit
class TestSimpleNames:
    """Tests for simple model name resolution."""

    def test_resolve_opus(self):
        """Resolve 'opus' to 'opus'."""
        assert resolve_model("opus") == "opus"

    def test_resolve_sonnet(self):
        """Resolve 'sonnet' to 'sonnet'."""
        assert resolve_model("sonnet") == "sonnet"

    def test_resolve_haiku(self):
        """Resolve 'haiku' to 'haiku'."""
        assert resolve_model("haiku") == "haiku"

    def test_resolve_case_insensitive_upper(self):
        """Simple names are case insensitive (uppercase)."""
        assert resolve_model("OPUS") == "opus"
        assert resolve_model("SONNET") == "sonnet"
        assert resolve_model("HAIKU") == "haiku"

    def test_resolve_case_insensitive_mixed(self):
        """Simple names are case insensitive (mixed case)."""
        assert resolve_model("Opus") == "opus"
        assert resolve_model("SoNnEt") == "sonnet"
        assert resolve_model("HaIkU") == "haiku"


@pytest.mark.unit
class TestOpenRouterSlugs:
    """Tests for OpenRouter slug resolution."""

    def test_resolve_claude_opus_45(self):
        """Resolve Claude Opus 4.5 slug."""
        assert resolve_model("anthropic/claude-opus-4.5") == "opus"

    def test_resolve_claude_sonnet_45(self):
        """Resolve Claude Sonnet 4.5 slug."""
        assert resolve_model("anthropic/claude-sonnet-4.5") == "sonnet"

    def test_resolve_claude_haiku_45(self):
        """Resolve Claude Haiku 4.5 slug."""
        assert resolve_model("anthropic/claude-haiku-4.5") == "haiku"

    def test_resolve_claude_opus_41(self):
        """Resolve Claude Opus 4.1 slug."""
        assert resolve_model("anthropic/claude-opus-4.1") == "opus"

    def test_resolve_claude_opus_4(self):
        """Resolve Claude Opus 4 slug."""
        assert resolve_model("anthropic/claude-opus-4") == "opus"

    def test_resolve_claude_sonnet_4(self):
        """Resolve Claude Sonnet 4 slug."""
        assert resolve_model("anthropic/claude-sonnet-4") == "sonnet"

    def test_resolve_claude_37_sonnet(self):
        """Resolve Claude 3.7 Sonnet slug."""
        assert resolve_model("anthropic/claude-3.7-sonnet") == "sonnet"

    def test_resolve_claude_35_haiku(self):
        """Resolve Claude 3.5 Haiku slug."""
        assert resolve_model("anthropic/claude-3.5-haiku") == "haiku"

    def test_resolve_claude_35_haiku_dated(self):
        """Resolve Claude 3.5 Haiku with date slug."""
        assert resolve_model("anthropic/claude-3.5-haiku-20241022") == "haiku"

    def test_resolve_claude_35_sonnet(self):
        """Resolve Claude 3.5 Sonnet slug."""
        assert resolve_model("anthropic/claude-3.5-sonnet") == "sonnet"

    def test_resolve_claude_35_sonnet_dated(self):
        """Resolve Claude 3.5 Sonnet with date slug."""
        assert resolve_model("anthropic/claude-3.5-sonnet-20240620") == "sonnet"

    def test_resolve_claude_3_series(self):
        """Resolve Claude 3 series slugs."""
        assert resolve_model("anthropic/claude-3-haiku") == "haiku"
        assert resolve_model("anthropic/claude-3-sonnet") == "sonnet"
        assert resolve_model("anthropic/claude-3-opus") == "opus"

    def test_resolve_case_insensitive_slug(self):
        """OpenRouter slugs are case insensitive."""
        assert resolve_model("ANTHROPIC/CLAUDE-SONNET-4") == "sonnet"
        assert resolve_model("Anthropic/Claude-Opus-4.5") == "opus"


@pytest.mark.unit
class TestUnsupportedModels:
    """Tests for unsupported model error handling."""

    def test_unknown_model_raises_error(self):
        """Unknown model raises UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError):
            resolve_model("gpt-4")

    def test_invalid_openrouter_format(self):
        """Invalid OpenRouter format raises error."""
        with pytest.raises(UnsupportedModelError):
            resolve_model("anthropic/gpt-4")

    def test_empty_string_raises_error(self):
        """Empty string raises error."""
        with pytest.raises(UnsupportedModelError):
            resolve_model("")

    def test_random_string_raises_error(self):
        """Random string raises error."""
        with pytest.raises(UnsupportedModelError):
            resolve_model("not-a-model")

    def test_partial_slug_raises_error(self):
        """Partial slug raises error."""
        with pytest.raises(UnsupportedModelError):
            resolve_model("anthropic/claude")

    def test_error_message_content(self):
        """Error message contains useful information."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            resolve_model("invalid-model")

        error = exc_info.value
        assert "invalid-model" in str(error)
        assert "opus" in str(error).lower() or "sonnet" in str(error).lower()
        assert error.model == "invalid-model"


@pytest.mark.unit
class TestMappingConsistency:
    """Tests for mapping configuration consistency."""

    def test_simple_names_set_complete(self):
        """Simple names set contains all expected values."""
        assert "opus" in SIMPLE_NAMES
        assert "sonnet" in SIMPLE_NAMES
        assert "haiku" in SIMPLE_NAMES
        assert len(SIMPLE_NAMES) == 3

    def test_available_models_have_slugs(self):
        """Available models list contains one entry per model family."""
        assert len(AVAILABLE_MODELS) == len(SIMPLE_NAMES)
        slugs = {m["slug"] for m in AVAILABLE_MODELS}
        for name in SIMPLE_NAMES:
            assert f"anthropic/claude-{name}" in slugs

    def test_available_models_format(self):
        """Available models have expected format and are resolvable."""
        for model in AVAILABLE_MODELS:
            assert model["slug"].startswith("anthropic/claude-")
            assert "Claude" in model["name"]
            assert resolve_model(model["slug"]) in SIMPLE_NAMES


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_whitespace_handling(self):
        """Leading/trailing whitespace: substring match still works."""
        assert resolve_model(" opus ") == "opus"

    def test_special_characters(self):
        """Special characters: substring match still works."""
        assert resolve_model("opus!") == "opus"
        assert resolve_model("sonnet@") == "sonnet"

    def test_unicode_characters(self):
        """Unicode characters: substring match still works."""
        assert resolve_model("opus™") == "opus"

    @pytest.mark.parametrize("model", list(SIMPLE_NAMES))
    def test_all_simple_names_resolvable(self, model):
        """Every simple name is resolvable."""
        result = resolve_model(model)
        assert result == model

"""Image format conversion utilities for OpenAI to Claude format."""

import re
from typing import Any

from .models import ContentPart, TextContent, ImageUrlContent


def parse_data_url(url: str) -> tuple[str, str]:
    """Extract media type and base64 data from a data URL.

    Args:
        url: Data URL in format data:image/xxx;base64,...

    Returns:
        Tuple of (media_type, base64_data)

    Raises:
        ValueError: If URL is not a valid data URL
    """
    match = re.match(r"data:([^;]+);base64,(.+)", url)
    if not match:
        raise ValueError(f"Invalid data URL format: {url[:50]}...")
    return match.group(1), match.group(2)


def is_http_url(url: str) -> bool:
    """Check if URL is an HTTP/HTTPS URL."""
    return url.startswith("http://") or url.startswith("https://")


def is_data_url(url: str) -> bool:
    """Check if URL is a data URL."""
    return url.startswith("data:")


def openai_image_to_claude(image_content: ImageUrlContent) -> dict[str, Any]:
    """Convert OpenAI image_url content block to Claude image format.

    OpenAI format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    Claude format:
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    """
    url = image_content.image_url.url

    if is_data_url(url):
        media_type, data = parse_data_url(url)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        }
    elif is_http_url(url):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            }
        }
    else:
        raise ValueError(f"Unsupported image URL format: {url[:50]}...")


def openai_content_to_claude(content: str | list[ContentPart]) -> list[dict[str, Any]]:
    """Convert OpenAI message content to Claude content array format.

    Args:
        content: Either a string or list of content parts (text/image_url)

    Returns:
        List of Claude-format content blocks
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    result = []
    for part in content:
        if isinstance(part, TextContent):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageUrlContent):
            result.append(openai_image_to_claude(part))
        else:
            # Handle dict-like objects (from Pydantic validation)
            if hasattr(part, "type"):
                if part.type == "text":
                    result.append({"type": "text", "text": part.text})
                elif part.type == "image_url":
                    result.append(openai_image_to_claude(part))

    return result


def has_multimodal_content(messages: list) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, ImageUrlContent) or (hasattr(part, "type") and part.type == "image_url"):
                    return True
    return False


def extract_text_from_content(content: str | list[ContentPart]) -> str:
    """Extract text from message content for logging.

    For multimodal content, replaces images with placeholder text.
    """
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if isinstance(part, TextContent) or (hasattr(part, "type") and part.type == "text"):
            parts.append(part.text)
        elif isinstance(part, ImageUrlContent) or (hasattr(part, "type") and part.type == "image_url"):
            url = part.image_url.url if hasattr(part, "image_url") else part.image_url["url"]
            if is_data_url(url):
                parts.append("[image: base64 data]")
            else:
                parts.append(f"[image: {url}]")

    return " ".join(parts)

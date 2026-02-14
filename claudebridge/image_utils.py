"""Image format conversion utilities for OpenAI to Claude format."""

import base64
import re
from dataclasses import dataclass
from typing import Any

from .models import ContentPart, TextContent, ImageUrlContent


EXTENSION_MAP = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


@dataclass
class AttachmentInfo:
    """Metadata for an attachment extracted from a message."""

    msg_index: int
    att_index: int
    media_type: str
    content_type: str  # "base64" or "url"
    data: bytes | None  # decoded binary data (base64 only)
    url: str | None  # original HTTP URL (url only)
    filename: str  # e.g. "msg0_att0.png"


def extract_attachments_from_messages(
    messages: list,
) -> list[AttachmentInfo]:
    """Extract attachment info from multimodal messages.

    For base64 data URLs: decodes to bytes, determines extension.
    For HTTP URLs: stores the URL string (no download).
    """
    attachments: list[AttachmentInfo] = []

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg.content, list):
            continue

        att_idx = 0
        for part in msg.content:
            if not isinstance(part, ImageUrlContent):
                continue

            url = part.image_url.url

            if is_data_url(url):
                media_type, b64_data = parse_data_url(url)
                ext = EXTENSION_MAP.get(media_type, ".bin")
                filename = f"msg{msg_idx}_att{att_idx}{ext}"
                attachments.append(
                    AttachmentInfo(
                        msg_index=msg_idx,
                        att_index=att_idx,
                        media_type=media_type,
                        content_type="base64",
                        data=base64.b64decode(b64_data),
                        url=None,
                        filename=filename,
                    )
                )
            elif is_http_url(url):
                # Guess extension from URL or default to .png
                ext = ".png"
                for mt, e in EXTENSION_MAP.items():
                    if e[1:] in url.lower():
                        ext = e
                        break
                filename = f"msg{msg_idx}_att{att_idx}{ext}"
                attachments.append(
                    AttachmentInfo(
                        msg_index=msg_idx,
                        att_index=att_idx,
                        media_type="image/unknown",
                        content_type="url",
                        data=None,
                        url=url,
                        filename=filename,
                    )
                )

            att_idx += 1

    return attachments


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
    """Convert OpenAI image_url content block to Claude image/document format.

    OpenAI format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,..."}}

    Claude format (images):
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}

    Claude format (PDFs):
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
    """
    url = image_content.image_url.url

    if is_data_url(url):
        media_type, data = parse_data_url(url)
        block_type = "document" if media_type == "application/pdf" else "image"
        return {
            "type": block_type,
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        }

    if is_http_url(url):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            }
        }

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

    return result


def has_multimodal_content(messages: list) -> bool:
    """Check if any message contains image content."""
    return any(
        isinstance(part, ImageUrlContent)
        for msg in messages
        if isinstance(msg.content, list)
        for part in msg.content
    )


def extract_text_from_content(content: str | list[ContentPart]) -> str:
    """Extract text from message content for logging.

    For multimodal content, replaces images/documents with placeholder text.
    """
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if isinstance(part, TextContent):
            parts.append(part.text)
        elif isinstance(part, ImageUrlContent):
            url = part.image_url.url
            if is_data_url(url):
                media_type, _ = parse_data_url(url)
                if media_type == "application/pdf":
                    parts.append("[document: PDF base64 data]")
                else:
                    parts.append("[image: base64 data]")
            else:
                parts.append(f"[image: {url}]")

    return " ".join(parts)

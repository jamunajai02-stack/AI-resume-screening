"""Tests for resume parser."""

import tempfile
from pathlib import Path

import pytest

from src.resume_parser import (
    EmptyDocumentError,
    UnsupportedFormatError,
    extract_resume_text,
)


def test_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"some text")
        path = f.name
    try:
        with pytest.raises(UnsupportedFormatError):
            extract_resume_text(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_file_not_found():
    with pytest.raises(Exception):
        extract_resume_text("/nonexistent/file.pdf")


def test_unsupported_format_from_bytes():
    from src.resume_parser import extract_text_from_bytes
    with pytest.raises(UnsupportedFormatError):
        extract_text_from_bytes(b"hello", "resume.txt")

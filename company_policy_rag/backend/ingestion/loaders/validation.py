"""Shared input limits and lossless text decoding for untrusted documents."""

from __future__ import annotations

import codecs
import zipfile
from pathlib import Path

from charset_normalizer import from_bytes

MAX_DOCUMENT_BYTES = 100 * 1024 * 1024
MAX_EXPANDED_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10000


def read_text(file_path: Path) -> str:
    data = file_path.read_bytes()
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 100MB limit.")
    encoding = "utf-8-sig"
    for marker, candidate in (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if data.startswith(marker):
            encoding = candidate
            break
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        matches = list(from_bytes(data))
        match = next(
            (
                candidate
                for candidate in matches
                if candidate.encoding == "cp1252"
                and candidate.percent_chaos <= (matches[0].percent_chaos + 10 if matches else 20)
            ),
            matches[0] if matches else None,
        )
        if match is None or match.encoding is None or match.percent_chaos > 20:
            raise ValueError("Text encoding could not be detected. Save as UTF-8 or Unicode.") from exc
        text = str(match)
    if any(ord(char) < 32 and char not in "\n\r\t\f" for char in text):
        raise ValueError("Binary or invalid control characters found in a text document.")
    return text


def validate_office_archive(file_path: Path, required_member: str) -> None:
    """Bound decompression before handing an OOXML package to its parser."""
    try:
        with zipfile.ZipFile(file_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                raise ValueError("Office document exceeds the expanded-content limit.")
            if required_member not in archive.namelist():
                raise ValueError("Document content does not match its Office file extension.")
            if any(m.flag_bits & 1 for m in members):
                raise ValueError("Encrypted Office documents are not supported. Upload an unlocked copy.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid Office document. Export a new DOCX, XLSX or PPTX copy.") from exc

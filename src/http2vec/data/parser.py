"""Robust parser for the CSIC 2010 raw HTTP request dumps.

The CSIC 2010 corpus stores many HTTP requests in a single flat text file. Each
request is a *block*:

* a request line ``METHOD TARGET HTTP/x.y`` (``TARGET`` is a full URL, possibly
  with a query string),
* header lines ``Name: Value`` until a blank line,
* for ``POST`` requests, a body line after that blank line (``GET`` has none).

Blocks are separated by blank lines. A new block is recognised purely by the
request line, so a body line never accidentally ends a request.

Security
--------
The payloads (targets, headers, bodies) are adversarial by design: SQL
injection, XSS, CRLF injection and similar. Everything here is treated as
**opaque text** - nothing is evaluated, executed, formatted-as-code or rendered.
Files are decoded explicitly as UTF-8 with ``errors="replace"`` so a malformed or
hostile file can never raise a surprising decoding error or smuggle in undecoded
bytes.
"""

from __future__ import annotations

import pathlib
import re

from .schemas import HttpRequest, Label

# A new request block starts at any line that looks like an HTTP request line.
_REQUEST_LINE_RE = re.compile(
    r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|CONNECT)\b.*\bHTTP/\d"
)


def _split_lines(text: str, *, strip_cr: bool) -> list[str]:
    """Split ``text`` on LF, optionally dropping a trailing CR from each line.

    Splitting only on ``"\\n"`` (instead of :meth:`str.splitlines`) avoids
    treating exotic Unicode separators that may occur inside adversarial payloads
    as line breaks. When ``strip_cr`` is true, CR and LF then act purely as line
    separators and never remain inside a stored line.
    """
    lines = text.split("\n")
    if strip_cr:
        return [line.rstrip("\r") for line in lines]
    return lines


def _strip_trailing_blanks(block_lines: list[str]) -> list[str]:
    """Drop trailing blank lines (the inter-request separators) from a block."""
    end = len(block_lines)
    while end > 0 and not block_lines[end - 1].strip():
        end -= 1
    return block_lines[:end]


def _parse_request_line(line: str) -> tuple[str, str, str]:
    """Split a request line into ``(method, target, version)``.

    Whitespace splitting yields at most three parts; a missing version degrades
    gracefully to an empty string rather than raising.
    """
    parts = line.split(maxsplit=2)
    method = parts[0] if len(parts) >= 1 else ""
    target = parts[1] if len(parts) >= 2 else ""
    version = parts[2] if len(parts) >= 3 else ""
    return method, target, version


def _parse_header(line: str) -> tuple[str, str]:
    """Split a header line on the first ``": "`` (falling back to ``":"``)."""
    if ": " in line:
        name, value = line.split(": ", 1)
    elif ":" in line:
        name, value = line.split(":", 1)
    else:
        return line.strip(), ""
    return name.strip(), value.strip()


def _parse_block(block_lines: list[str], label: Label) -> HttpRequest:
    """Turn one accumulated block into an :class:`HttpRequest`."""
    method, target, version = _parse_request_line(block_lines[0])

    blank_index: int | None = None
    for index in range(1, len(block_lines)):
        if not block_lines[index].strip():
            blank_index = index
            break

    if blank_index is None:
        header_lines = block_lines[1:]
        body = ""
    else:
        header_lines = block_lines[1:blank_index]
        body = "\n".join(
            line for line in block_lines[blank_index + 1 :] if line.strip()
        )

    headers = tuple(_parse_header(line) for line in header_lines)
    raw = "\n".join(block_lines)
    return HttpRequest(
        method=method,
        target=target,
        version=version,
        headers=headers,
        body=body,
        label=label,
        raw=raw,
    )


def parse_csic_text(
    text: str, label: Label, *, encode_crlf_literally: bool = True
) -> list[HttpRequest]:
    """Parse a CSIC 2010 text dump into a list of :class:`HttpRequest`.

    Args:
        text: The full contents of a CSIC traffic file.
        label: The label assigned to every request in ``text``.
        encode_crlf_literally: When ``True`` (the default, documented behaviour)
            CR and LF are treated strictly as line separators: a trailing CR is
            stripped from each line, so no CR/LF ever appears inside a stored
            field. The flag is part of the public contract for API
            compatibility; the documented behaviour is the only supported one.

    Returns:
        The parsed requests, in file order.
    """
    lines = _split_lines(text, strip_cr=encode_crlf_literally)

    requests: list[HttpRequest] = []
    current: list[str] = []
    for line in lines:
        if _REQUEST_LINE_RE.match(line):
            if current:
                requests.append(_parse_block(_strip_trailing_blanks(current), label))
            current = [line]
        elif current:
            current.append(line)
    if current:
        requests.append(_parse_block(_strip_trailing_blanks(current), label))
    return requests


def parse_csic_file(
    path: pathlib.Path, label: Label, *, encode_crlf_literally: bool = True
) -> list[HttpRequest]:
    """Parse a CSIC 2010 file from disk. See :func:`parse_csic_text`.

    The file is read with an explicit UTF-8 decoding (``errors="replace"``) so
    adversarial or malformed bytes cannot raise a decoding error.
    """
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_csic_text(text, label, encode_crlf_literally=encode_crlf_literally)

"""ZScaler Web Proxy log parser.

ZScaler NSS feeds are admin-configurable, so the column order varies between
deployments. This parser is tolerant: it inspects the first row, and if it
looks like a header it builds a mapping from header names to our canonical
fields via FIELD_ALIASES. Otherwise it falls back to DEFAULT_FIELD_ORDER
(a common ZScaler web log layout).

CSV and TSV are both supported; the delimiter is sniffed.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timezone
from typing import Iterable

CANONICAL_FIELDS = {
    "timestamp",
    "source_ip",
    "user_agent",
    "action",
    "url",
    "method",
    "status_code",
    "bytes_sent",
    "url_category",
    "threat_name",
    "user_login",
    # The CSIC training data carries `content_length` as a raw
    # "Content-Length: N" string (or empty). The ML preprocessor reads
    # that exact shape, so we preserve it verbatim end-to-end.
    "content_length",
}

FIELD_ALIASES: dict[str, str] = {
    # timestamp
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "logtime": "timestamp",
    # source ip
    "sourceip": "source_ip",
    "source_ip": "source_ip",
    "clientip": "source_ip",
    "client_ip": "source_ip",
    "cip": "source_ip",
    # user agent
    "useragent": "user_agent",
    "user_agent": "user_agent",
    "ua": "user_agent",
    # action
    "action": "action",
    # url
    "url": "url",
    "urlhost": "url",
    "host": "url",
    "fullurl": "url",
    # method
    "requestmethod": "method",
    "method": "method",
    "httpmethod": "method",
    # status
    "status": "status_code",
    "statuscode": "status_code",
    "responsecode": "status_code",
    "http_status": "status_code",
    # bytes
    "responsesize": "bytes_sent",
    "bytes": "bytes_sent",
    "bytes_sent": "bytes_sent",
    "rsp_size": "bytes_sent",
    # category
    "urlcategory": "url_category",
    "category": "url_category",
    # threat
    "threatname": "threat_name",
    "threat": "threat_name",
    "threatclass": "threat_name",
    # user
    "user": "user_login",
    "login": "user_login",
    "username": "user_login",
    # content length (raw header string, as the CSIC training data carries)
    "content_length": "content_length",
    "contentlength": "content_length",
}

# Used when there is no header row.
DEFAULT_FIELD_ORDER: list[str] = [
    "timestamp",
    "user_login",
    "url",
    "url_category",
    "action",
    "threat_name",
    "user_agent",
    "source_ip",
    "method",
    "bytes_sent",
    "status_code",
]

_TS_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%a %b %d %H:%M:%S %Y",
    "%d/%b/%Y:%H:%M:%S %z",
    "%Y/%m/%d %H:%M:%S",
]


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    s = value.strip().strip('"')
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Last resort: ISO 8601 fromisoformat.
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None

def _parse_content_length(value: str | None) -> float | None:
    if not value:
        return None
    try:
        pattern = "\d+"
        match = re.search(pattern=pattern, string=value)
        if match:
            num = float(match.group(0))
            return num
        else:
            return 0.0
    except Exception as e:
        return None

def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = value.strip()
    if not s or s in {"-", "NA", "null", "None"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _looks_like_header(row: list[str]) -> bool:
    if not row:
        return False
    normalized = [_normalize_header(c) for c in row]
    return sum(1 for c in normalized if c in FIELD_ALIASES) >= 2

def _sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t|")
    except csv.Error:
        # Default to CSV.
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _D()

# INVESTIGATION STEP 7: ZSCALER log string is fed to this function
def parse_zscaler_log(content: str) -> tuple[list[dict], int]:
    """Parse a ZScaler web proxy log.

    Returns (entries, skipped). Each entry contains the canonical fields plus
    `raw_line`. Lines that cannot be parsed at all are counted in `skipped`.
    """
    if not content.strip():
        return [], 0

    sample = "\n".join(content.splitlines()[:5])
    dialect = _sniff_dialect(sample)

    reader = csv.reader(io.StringIO(content), dialect=dialect)
    rows: Iterable[list[str]] = (r for r in reader if r)

    try:
        first = next(rows)
    except StopIteration:
        return [], 0

    if _looks_like_header(first):
        headers = [_normalize_header(c) for c in first]
        field_for_col = [FIELD_ALIASES.get(h) for h in headers]
    else:
        # Treat first row as data; use the default field order.
        field_for_col = list(DEFAULT_FIELD_ORDER)
        rows = _prepend(first, rows)

    entries: list[dict] = []
    skipped = 0

    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        try:
            entry = _row_to_entry(row, field_for_col)
        except Exception:
            skipped += 1
            continue
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)

    return entries, skipped


def _prepend(item, iterator):
    yield item
    yield from iterator


def _row_to_entry(row: list[str], field_for_col: list[str | None]) -> dict | None:
    raw_line = ",".join(row)
    entry: dict = {f: None for f in CANONICAL_FIELDS}
    entry["raw_line"] = raw_line

    for idx, value in enumerate(row):
        if idx >= len(field_for_col):
            break
        canonical = field_for_col[idx]
        if not canonical:
            continue
        cleaned = value.strip()
        if not cleaned or cleaned == "-":
            continue
        entry[canonical] = cleaned

    entry["timestamp"] = _parse_timestamp(entry.get("timestamp"))
    entry["status_code"] = _to_int(entry.get("status_code"))
    entry["bytes_sent"] = _to_int(entry.get("bytes_sent"))
    entry["content_length"] = _parse_content_length(entry.get("content_length"))

    if entry.get("bytes_sent") is None:
        cl = entry.get("content_length")
        entry["bytes_sent"] = cl

    has_anything = any(
        entry.get(k)
        for k in ("timestamp", "source_ip", "url", "user_login", "action")
    )
    if not has_anything:
        return None
    return entry

if __name__ == "__main__":
    filepath = r"/Users/nbusich/Documents/coding/tenex/src/nextjs-frontend/public/samples/csic-attack-1.csv"
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    entries, skipped = parse_zscaler_log(content)

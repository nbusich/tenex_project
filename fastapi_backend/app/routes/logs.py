from collections import Counter
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.anomaly import detect_anomalies, explain_anomalies, explain_entry
from app.anomaly.detector import ALLOWED_MODELS, DEFAULT_MODEL
from app.database import User, get_async_session
from app.log_parsing import parse_zscaler_log
from app.models import LogEntry, LogFile
from app.schemas import (
    LogEntryRead,
    LogFileRead,
    LogFileSummary,
    TimelineBucket,
    TopIP,
    UploadResponse,
)
from app.users import current_active_user

router = APIRouter(tags=["logs"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
TIMELINE_BUCKETS = 24


def _entry_dict_to_model(entry: dict, log_file_id: UUID) -> LogEntry:
    return LogEntry(
        log_file_id=log_file_id,
        timestamp=entry.get("timestamp"),
        source_ip=entry.get("source_ip"),
        user_agent=entry.get("user_agent"),
        action=entry.get("action"),
        url=entry.get("url"),
        method=entry.get("method"),
        status_code=entry.get("status_code"),
        bytes_sent=entry.get("bytes_sent"),
        url_category=entry.get("url_category"),
        threat_name=entry.get("threat_name"),
        user_login=entry.get("user_login"),
        raw_line=entry.get("raw_line"),
        is_anomaly=bool(entry.get("is_anomaly")),
        anomaly_score=entry.get("anomaly_score"),
        anomaly_reason=entry.get("anomaly_reason"),
    )


# INVESTIGATION STEP 4: The API hands the logs and model to the correct route. The route is a function.
@router.post("/upload", response_model=UploadResponse)
async def upload_log_file(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    chosen_model = (model or DEFAULT_MODEL).strip().lower()
    if chosen_model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Allowed: {', '.join(ALLOWED_MODELS)}.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB).")
    try:
        # INVESTIGATION STEP 5: The raw log bytes are processed to a string
        content = raw.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Could not decode: {exc}")
    
    # INVESTIGATION STEP 6: The log string is handed to parse_zscaler_log, converts string to list of log entries (dicts)
    entries, skipped = parse_zscaler_log(content)
    if not entries:
        raise HTTPException(
            status_code=400,
            detail="No parseable entries found. Is this a ZScaler web proxy log?",
        )
    
    # INVESTIGATION STEP 7: list of log entries, each one is a dict with field/value
    entries = detect_anomalies(entries, model_name=chosen_model)
    anomaly_count = sum(1 for e in entries if e.get("is_anomaly"))
    
    
    log_file = LogFile(
        filename=file.filename or "uploaded.log",
        user_id=user.id,
        total_entries=len(entries),
        anomaly_count=anomaly_count,
    )
    # INVESTIGATION STEP 14: Results and log added to database
    db.add(log_file)
    await db.flush()  # populate log_file.id

    db.add_all([_entry_dict_to_model(e, log_file.id) for e in entries])
    await db.commit()
    await db.refresh(log_file)

    return UploadResponse(
        file=LogFileRead.model_validate(log_file),
        parsed=len(entries),
        skipped=skipped,
        anomalies=anomaly_count,
    )


def _transform_files(files):
    return [LogFileRead.model_validate(f) for f in files]


def _transform_entries(rows):
    return [LogEntryRead.model_validate(r) for r in rows]


@router.get("/files", response_model=Page[LogFileRead])
async def list_log_files(
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
):
    params = Params(page=page, size=size)
    query = (
        select(LogFile)
        .filter(LogFile.user_id == user.id)
        .order_by(LogFile.uploaded_at.desc())
    )
    return await apaginate(db, query, params, transformer=_transform_files)


async def _load_owned_file(file_id: UUID, db: AsyncSession, user: User) -> LogFile:
    result = await db.execute(
        select(LogFile).filter(LogFile.id == file_id, LogFile.user_id == user.id)
    )
    log_file = result.scalars().first()
    if not log_file:
        raise HTTPException(status_code=404, detail="Log file not found.")
    return log_file


@router.get("/files/{file_id}/entries", response_model=Page[LogEntryRead])
async def list_log_entries(
    file_id: UUID,
    only_anomalies: bool = Query(False),
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
    page: int = Query(1, ge=1),
    size: int = Query(25, ge=1, le=200),
):
    await _load_owned_file(file_id, db, user)
    params = Params(page=page, size=size)
    query = select(LogEntry).filter(LogEntry.log_file_id == file_id)
    if only_anomalies:
        query = query.filter(LogEntry.is_anomaly.is_(True))
    query = query.order_by(LogEntry.timestamp.asc().nullslast())
    return await apaginate(db, query, params, transformer=_transform_entries)


@router.get("/files/{file_id}/summary", response_model=LogFileSummary)
async def get_log_file_summary(
    file_id: UUID,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    log_file = await _load_owned_file(file_id, db, user)

    result = await db.execute(
        select(LogEntry).filter(LogEntry.log_file_id == file_id)
    )
    entries = list(result.scalars().all())

    timeline = _build_timeline(entries)
    top_ips = _build_top_ips(entries)
    top_actions = _top_n_counter(
        Counter((e.action or "unknown") for e in entries), n=5
    )
    top_categories = _top_n_counter(
        Counter((e.url_category or "uncategorized") for e in entries), n=5
    )

    anomaly_dicts = [
        {
            "timestamp": e.timestamp,
            "source_ip": e.source_ip,
            "url": e.url,
            "action": e.action,
            "status_code": e.status_code,
            "threat_name": e.threat_name,
            "anomaly_score": e.anomaly_score,
            "anomaly_reason": e.anomaly_reason,
        }
        for e in entries
        if e.is_anomaly
    ]

    return LogFileSummary(
        file=LogFileRead.model_validate(log_file),
        timeline=timeline,
        top_source_ips=top_ips,
        top_actions=top_actions,
        top_categories=top_categories,
        ai_explanation=explain_anomalies(anomaly_dicts),
    )


@router.delete("/files/{file_id}")
async def delete_log_file(
    file_id: UUID,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    log_file = await _load_owned_file(file_id, db, user)
    await db.delete(log_file)
    await db.commit()
    return {"message": "Log file deleted."}


@router.post("/entries/{entry_id}/explain")
async def explain_log_entry(
    entry_id: UUID,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """LLM-backed per-entry verdict.

    Returns `{"verdict": "anomaly" | "false_positive" | "normal",
              "description": "...",
              "source": "llm" | "fallback"}`.

    Authorizes via the parent log file's ownership.
    """
    result = await db.execute(
        select(LogEntry)
        .join(LogFile, LogEntry.log_file_id == LogFile.id)
        .filter(LogEntry.id == entry_id, LogFile.user_id == user.id)
    )
    entry = result.scalars().first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found.")

    payload = {
        "timestamp": entry.timestamp,
        "source_ip": entry.source_ip,
        "user_login": entry.user_login,
        "method": entry.method,
        "url": entry.url,
        "status_code": entry.status_code,
        "action": entry.action,
        "threat_name": entry.threat_name,
        "url_category": entry.url_category,
        "anomaly_score": entry.anomaly_score,
        "anomaly_reason": entry.anomaly_reason,
        "is_anomaly": entry.is_anomaly,
    }
    return explain_entry(payload)


def _build_timeline(entries: list[LogEntry]) -> list[TimelineBucket]:
    timed = [e for e in entries if e.timestamp]
    if not timed:
        return []
    times = [e.timestamp for e in timed]
    start = min(times)
    end = max(times)
    span = (end - start).total_seconds()
    if span <= 0:
        return [
            TimelineBucket(
                bucket=start,
                count=len(timed),
                anomaly_count=sum(1 for e in timed if e.is_anomaly),
            )
        ]
    bucket_seconds = span / TIMELINE_BUCKETS
    buckets: dict[int, dict] = {}
    for e in timed:
        idx = int((e.timestamp - start).total_seconds() / bucket_seconds)
        if idx >= TIMELINE_BUCKETS:
            idx = TIMELINE_BUCKETS - 1
        b = buckets.setdefault(idx, {"count": 0, "anomaly_count": 0})
        b["count"] += 1
        if e.is_anomaly:
            b["anomaly_count"] += 1
    out: list[TimelineBucket] = []
    for i in range(TIMELINE_BUCKETS):
        b = buckets.get(i)
        if not b:
            continue
        bucket_ts = datetime.fromtimestamp(
            start.timestamp() + i * bucket_seconds, tz=start.tzinfo
        )
        out.append(
            TimelineBucket(
                bucket=bucket_ts,
                count=b["count"],
                anomaly_count=b["anomaly_count"],
            )
        )
    return out


def _build_top_ips(entries: list[LogEntry], limit: int = 5) -> list[TopIP]:
    counts: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    for e in entries:
        if not e.source_ip:
            continue
        counts[e.source_ip] += 1
        if e.is_anomaly:
            anomaly_counts[e.source_ip] += 1
    return [
        TopIP(source_ip=ip, count=c, anomaly_count=anomaly_counts[ip])
        for ip, c in counts.most_common(limit)
    ]


def _top_n_counter(counter: Counter, n: int = 5) -> dict[str, int]:
    return {k: v for k, v in counter.most_common(n)}

import uuid
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi_users import schemas
from pydantic import BaseModel


class UserRead(schemas.BaseUser[uuid.UUID]):
    pass


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


class ItemBase(BaseModel):
    name: str
    description: str | None = None
    quantity: int | None = None


class ItemCreate(ItemBase):
    pass


class ItemRead(ItemBase):
    id: UUID
    user_id: UUID

    model_config = {"from_attributes": True}


class LogFileRead(BaseModel):
    id: UUID
    filename: str
    uploaded_at: datetime
    total_entries: int
    anomaly_count: int

    model_config = {"from_attributes": True}


class LogEntryRead(BaseModel):
    id: UUID
    log_file_id: UUID
    timestamp: Optional[datetime] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    action: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    bytes_sent: Optional[int] = None
    url_category: Optional[str] = None
    threat_name: Optional[str] = None
    user_login: Optional[str] = None
    raw_line: Optional[str] = None
    is_anomaly: bool
    anomaly_score: Optional[float] = None
    anomaly_reason: Optional[str] = None

    model_config = {"from_attributes": True}


class TimelineBucket(BaseModel):
    bucket: datetime
    count: int
    anomaly_count: int


class TopIP(BaseModel):
    source_ip: str
    count: int
    anomaly_count: int


class LogFileSummary(BaseModel):
    file: LogFileRead
    timeline: list[TimelineBucket]
    top_source_ips: list[TopIP]
    top_actions: dict[str, int]
    top_categories: dict[str, int]
    ai_explanation: Optional[str] = None


class UploadResponse(BaseModel):
    file: LogFileRead
    parsed: int
    skipped: int
    anomalies: int

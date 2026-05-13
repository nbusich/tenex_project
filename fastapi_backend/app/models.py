from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Boolean,
    Float,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from uuid import uuid4


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    items = relationship("Item", back_populates="user", cascade="all, delete-orphan")
    log_files = relationship(
        "LogFile", back_populates="user", cascade="all, delete-orphan"
    )


class Item(Base):
    __tablename__ = "items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    quantity = Column(Integer, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)

    user = relationship("User", back_populates="items")


class LogFile(Base):
    __tablename__ = "log_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    total_entries = Column(Integer, nullable=False, default=0)
    anomaly_count = Column(Integer, nullable=False, default=0)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)

    user = relationship("User", back_populates="log_files")
    entries = relationship(
        "LogEntry", back_populates="log_file", cascade="all, delete-orphan"
    )


class LogEntry(Base):
    __tablename__ = "log_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    log_file_id = Column(
        UUID(as_uuid=True), ForeignKey("log_files.id"), nullable=False, index=True
    )

    timestamp = Column(DateTime(timezone=True), nullable=True, index=True)
    source_ip = Column(String, nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    action = Column(String, nullable=True)

    url = Column(Text, nullable=True)
    method = Column(String, nullable=True)
    status_code = Column(Integer, nullable=True)
    bytes_sent = Column(BigInteger, nullable=True)
    url_category = Column(String, nullable=True)
    threat_name = Column(String, nullable=True)
    user_login = Column(String, nullable=True)

    raw_line = Column(Text, nullable=True)

    is_anomaly = Column(Boolean, nullable=False, default=False, index=True)
    anomaly_score = Column(Float, nullable=True)
    anomaly_reason = Column(Text, nullable=True)

    log_file = relationship("LogFile", back_populates="entries")

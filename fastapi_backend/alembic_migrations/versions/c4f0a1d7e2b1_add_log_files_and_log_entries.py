"""Add log_files and log_entries

Revision ID: c4f0a1d7e2b1
Revises: b389592974f8
Create Date: 2026-05-07 20:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4f0a1d7e2b1"
down_revision: Union[str, None] = "b389592974f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "log_files",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("total_entries", sa.Integer(), nullable=False),
        sa.Column("anomaly_count", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "log_entries",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("log_file_id", sa.UUID(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_ip", sa.String(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("action", sa.String(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("method", sa.String(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("bytes_sent", sa.BigInteger(), nullable=True),
        sa.Column("url_category", sa.String(), nullable=True),
        sa.Column("threat_name", sa.String(), nullable=True),
        sa.Column("user_login", sa.String(), nullable=True),
        sa.Column("raw_line", sa.Text(), nullable=True),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("anomaly_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["log_file_id"], ["log_files.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_log_entries_log_file_id", "log_entries", ["log_file_id"], unique=False
    )
    op.create_index(
        "ix_log_entries_timestamp", "log_entries", ["timestamp"], unique=False
    )
    op.create_index(
        "ix_log_entries_source_ip", "log_entries", ["source_ip"], unique=False
    )
    op.create_index(
        "ix_log_entries_is_anomaly", "log_entries", ["is_anomaly"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_log_entries_is_anomaly", table_name="log_entries")
    op.drop_index("ix_log_entries_source_ip", table_name="log_entries")
    op.drop_index("ix_log_entries_timestamp", table_name="log_entries")
    op.drop_index("ix_log_entries_log_file_id", table_name="log_entries")
    op.drop_table("log_entries")
    op.drop_table("log_files")

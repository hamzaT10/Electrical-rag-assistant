"""extend ingestion jobs for worker

Revision ID: 7b4c0bb0e8a5
Revises: 5d05177ed4e4
Create Date: 2026-06-24 09:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b4c0bb0e8a5"
down_revision: str | Sequence[str] | None = "5d05177ed4e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("ingestion_jobs", sa.Column("source_path", sa.String(length=1024), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("task_id", sa.String(length=255), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("pdf_files", sa.Integer(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("pages_loaded", sa.Integer(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("pages_ocr", sa.Integer(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("chunks_created", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_ingestion_jobs_task_id"), "ingestion_jobs", ["task_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_ingestion_jobs_task_id"), table_name="ingestion_jobs")
    op.drop_column("ingestion_jobs", "chunks_created")
    op.drop_column("ingestion_jobs", "pages_ocr")
    op.drop_column("ingestion_jobs", "pages_loaded")
    op.drop_column("ingestion_jobs", "pdf_files")
    op.drop_column("ingestion_jobs", "task_id")
    op.drop_column("ingestion_jobs", "source_path")

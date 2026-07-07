"""add document level ingestion

Revision ID: 91f4c0ac933b
Revises: 7b4c0bb0e8a5
Create Date: 2026-06-25 09:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "91f4c0ac933b"
down_revision: str | Sequence[str] | None = "7b4c0bb0e8a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("documents", sa.Column("storage_path", sa.String(length=1024), nullable=True))
    op.add_column("documents", sa.Column("indexed_at", sa.DateTime(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("document_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_ingestion_jobs_document_id_documents",
        "ingestion_jobs",
        "documents",
        ["document_id"],
        ["id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_ingestion_jobs_document_id_documents",
        "ingestion_jobs",
        type_="foreignkey",
    )
    op.drop_column("ingestion_jobs", "document_id")
    op.drop_column("documents", "indexed_at")
    op.drop_column("documents", "storage_path")

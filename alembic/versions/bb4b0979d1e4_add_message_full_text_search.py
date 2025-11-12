"""add_message_full_text_search

Revision ID: bb4b0979d1e4
Revises: afcaed29bdd7
Create Date: 2025-11-07 06:32:04.155970

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bb4b0979d1e4'
down_revision: Union[str, Sequence[str], None] = 'afcaed29bdd7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add full-text search column and GIN index to messages table."""
    # Add generated tsvector column for full-text search
    # GENERATED ALWAYS AS STORED means:
    # - Automatically computed on INSERT/UPDATE
    # - Stored in the table (not computed on query)
    # - Uses English text search configuration
    # - COALESCE handles NULL content gracefully
    op.execute("""
        ALTER TABLE messages
        ADD COLUMN content_search tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', COALESCE(content, ''))
        ) STORED
    """)

    # Create GIN (Generalized Inverted Index) for fast full-text search
    # GIN indexes are optimized for tsvector columns
    # Provides O(log n) search time instead of O(n) table scans
    op.create_index(
        'idx_messages_content_search',
        'messages',
        ['content_search'],
        postgresql_using='gin',
    )


def downgrade() -> None:
    """Remove full-text search column and index from messages table."""
    # Drop index first (required before dropping column)
    op.drop_index('idx_messages_content_search', table_name='messages')

    # Drop generated column
    op.drop_column('messages', 'content_search')

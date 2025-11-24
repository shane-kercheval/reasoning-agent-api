"""
add_total_cost_and_remove_tool_calls.

Revision ID: afcaed29bdd7
Revises: 6e688aaae9c9
Create Date: 2025-11-07 05:47:22.037099

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'afcaed29bdd7'
down_revision: str | Sequence[str] | None = '6e688aaae9c9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add total_cost column and remove tool_calls column."""
    # Add total_cost column (nullable initially for backfill)
    op.add_column('messages', sa.Column('total_cost', sa.Numeric(precision=10, scale=6), nullable=True))

    # Create index for cost queries
    op.create_index('idx_messages_total_cost', 'messages', ['total_cost'])

    # Backfill existing data from metadata->cost->total_cost
    op.execute("""
        UPDATE messages
        SET total_cost = CAST(metadata->'cost'->>'total_cost' AS NUMERIC)
        WHERE metadata->'cost'->>'total_cost' IS NOT NULL
    """)

    # Remove tool_calls column (not being used, redundant with reasoning_events)
    op.drop_column('messages', 'tool_calls')


def downgrade() -> None:
    """Remove total_cost column and restore tool_calls column."""
    # Restore tool_calls column
    op.add_column('messages', sa.Column('tool_calls', JSONB(), nullable=True))

    # Remove total_cost index and column
    op.drop_index('idx_messages_total_cost', table_name='messages')
    op.drop_column('messages', 'total_cost')

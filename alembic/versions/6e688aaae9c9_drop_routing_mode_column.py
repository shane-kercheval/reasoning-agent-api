"""drop_routing_mode_column

Revision ID: 6e688aaae9c9
Revises: 6904aae0a5a8
Create Date: 2025-11-01 13:50:39.535586

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6e688aaae9c9'
down_revision: Union[str, Sequence[str], None] = '6904aae0a5a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop routing_mode column from conversations table."""
    op.drop_column('conversations', 'routing_mode')


def downgrade() -> None:
    """Add routing_mode column back to conversations table."""
    op.add_column(
        'conversations',
        sa.Column('routing_mode', sa.String(), nullable=True)
    )

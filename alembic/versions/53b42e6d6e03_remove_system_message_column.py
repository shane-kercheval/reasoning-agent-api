"""remove_system_message_column

Revision ID: 53b42e6d6e03
Revises: bb4b0979d1e4
Create Date: 2025-11-17 22:14:45.841474

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '53b42e6d6e03'
down_revision: Union[str, Sequence[str], None] = 'bb4b0979d1e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop system_message column from conversations table."""
    op.drop_column('conversations', 'system_message')


def downgrade() -> None:
    """Restore system_message column to conversations table."""
    op.add_column(
        'conversations',
        sa.Column(
            'system_message',
            sa.Text(),
            nullable=False,
            server_default='You are a helpful assistant.'
        )
    )

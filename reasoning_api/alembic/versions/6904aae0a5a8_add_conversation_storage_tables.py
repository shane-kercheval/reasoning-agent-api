"""
Add conversation storage tables.

Revision ID: 6904aae0a5a8
Revises:
Create Date: 2025-10-26 17:00:55.122543

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '6904aae0a5a8'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create conversations and messages tables for persistent conversation storage."""
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.UUID(), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('system_message', sa.Text(), nullable=False, server_default='You are a helpful assistant.'),
        sa.Column('routing_mode', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('archived_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for conversations table
    op.create_index('idx_conversations_user_id', 'conversations', ['user_id'])
    op.create_index('idx_conversations_created_at', 'conversations', ['created_at'], postgresql_ops={'created_at': 'DESC'})

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('conversation_id', sa.UUID(), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('reasoning_events', JSONB(), nullable=True),
        sa.Column('tool_calls', JSONB(), nullable=True),
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('sequence_number', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('conversation_id', 'sequence_number', name='unique_conversation_sequence'),
    )

    # Create indexes for messages table
    op.create_index('idx_messages_conversation_id', 'messages', ['conversation_id', 'sequence_number'])


def downgrade() -> None:
    """Drop conversations and messages tables."""
    # Drop tables in reverse order (messages first due to foreign key)
    op.drop_index('idx_messages_conversation_id', table_name='messages')
    op.drop_table('messages')

    op.drop_index('idx_conversations_created_at', table_name='conversations')
    op.drop_index('idx_conversations_user_id', table_name='conversations')
    op.drop_table('conversations')

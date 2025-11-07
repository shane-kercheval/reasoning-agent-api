/**
 * Tests for ConversationList component.
 *
 * Tests rendering, loading states, empty state, and user interactions.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { ConversationList } from '../../src/components/conversations/ConversationList';
import type { ConversationSummary } from '../../src/lib/api-client';

// Mock data
const mockConversations: ConversationSummary[] = [
  {
    id: 'conv-1',
    title: 'First Conversation',
    system_message: 'You are helpful',
    created_at: '2024-01-01T10:00:00Z',
    updated_at: '2024-01-01T10:30:00Z',
    archived_at: null,
    message_count: 5,
  },
  {
    id: 'conv-2',
    title: null,
    system_message: 'You are helpful',
    created_at: '2024-01-02T10:00:00Z',
    updated_at: '2024-01-02T10:30:00Z',
    archived_at: null,
    message_count: 3,
  },
];

describe('ConversationList', () => {
  const defaultProps = {
    conversations: mockConversations,
    selectedConversationId: null,
    isLoading: false,
    error: null,
    viewFilter: 'active' as const,
    searchQuery: '',
    onSelectConversation: jest.fn(),
    onNewConversation: jest.fn(),
    onDeleteConversation: jest.fn().mockResolvedValue(undefined),
    onArchiveConversation: jest.fn().mockResolvedValue(undefined),
    onUpdateTitle: jest.fn().mockResolvedValue(undefined),
    onRefresh: jest.fn(),
    onViewFilterChange: jest.fn(),
    onSearchQueryChange: jest.fn(),
    onSearch: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders header with title', () => {
      render(<ConversationList {...defaultProps} />);

      expect(screen.getByText('Conversations')).toBeInTheDocument();
    });

    it('renders new conversation button', () => {
      render(<ConversationList {...defaultProps} />);

      expect(screen.getByText('New Conversation')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      render(<ConversationList {...defaultProps} />);

      const refreshButton = screen.getByTitle('Refresh conversations');
      expect(refreshButton).toBeInTheDocument();
    });

    it('renders all conversations', () => {
      render(<ConversationList {...defaultProps} />);

      expect(screen.getByText('First Conversation')).toBeInTheDocument();
      expect(screen.getByText('Untitled')).toBeInTheDocument();
    });

    it('shows message count for each conversation', () => {
      render(<ConversationList {...defaultProps} />);

      expect(screen.getByText(/5 messages/)).toBeInTheDocument();
      expect(screen.getByText(/3 messages/)).toBeInTheDocument();
    });

    it('highlights selected conversation', () => {
      render(<ConversationList {...defaultProps} selectedConversationId="conv-1" />);

      const conversationItems = screen.getAllByRole('generic').filter(
        (el) => el.className.includes('bg-primary/10'),
      );
      expect(conversationItems.length).toBeGreaterThan(0);
    });
  });

  describe('empty state', () => {
    it('shows empty state when no conversations and not loading', () => {
      render(<ConversationList {...defaultProps} conversations={[]} />);

      expect(screen.getByText('No conversations yet')).toBeInTheDocument();
      expect(screen.getByText('Start a new conversation to begin')).toBeInTheDocument();
    });

    it('does not show empty state when loading', () => {
      render(<ConversationList {...defaultProps} conversations={[]} isLoading={true} />);

      expect(screen.queryByText('No conversations yet')).not.toBeInTheDocument();
    });
  });

  describe('loading states', () => {
    it('shows initial loading spinner when no conversations loaded', () => {
      render(<ConversationList {...defaultProps} conversations={[]} isLoading={true} />);

      expect(screen.getByText('Loading conversations...')).toBeInTheDocument();
      expect(screen.getByText('Please wait')).toBeInTheDocument();
    });

    it('shows refresh overlay when refreshing with existing conversations', () => {
      render(<ConversationList {...defaultProps} isLoading={true} />);

      expect(screen.getByText('Refreshing...')).toBeInTheDocument();
      // Should still show conversations in background
      expect(screen.getByText('First Conversation')).toBeInTheDocument();
    });

    it('disables refresh button when loading', () => {
      render(<ConversationList {...defaultProps} isLoading={true} />);

      const refreshButton = screen.getByTitle('Refresh conversations');
      expect(refreshButton).toBeDisabled();
    });
  });

  describe('error state', () => {
    it('displays error message', () => {
      render(<ConversationList {...defaultProps} error="Failed to load conversations" />);

      expect(screen.getByText('Error loading conversations')).toBeInTheDocument();
      expect(screen.getByText('Failed to load conversations')).toBeInTheDocument();
    });

    it('shows error alongside conversations', () => {
      render(<ConversationList {...defaultProps} error="Network error" />);

      expect(screen.getByText('Error loading conversations')).toBeInTheDocument();
      expect(screen.getByText('First Conversation')).toBeInTheDocument();
    });
  });

  describe('user interactions', () => {
    it('calls onNewConversation when new button clicked', () => {
      render(<ConversationList {...defaultProps} />);

      const newButton = screen.getByText('New Conversation');
      fireEvent.click(newButton);

      expect(defaultProps.onNewConversation).toHaveBeenCalledTimes(1);
    });

    it('calls onRefresh when refresh button clicked', () => {
      render(<ConversationList {...defaultProps} />);

      const refreshButton = screen.getByTitle('Refresh conversations');
      fireEvent.click(refreshButton);

      expect(defaultProps.onRefresh).toHaveBeenCalledTimes(1);
    });

    it('calls onSelectConversation when conversation clicked', () => {
      render(<ConversationList {...defaultProps} />);

      const conversation = screen.getByText('First Conversation');
      fireEvent.click(conversation);

      expect(defaultProps.onSelectConversation).toHaveBeenCalledWith('conv-1');
    });
  });
});

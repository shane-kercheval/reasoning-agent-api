/**
 * Tests for ConversationItem component.
 *
 * Tests inline editing, delete functionality, and display formatting.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ConversationItem } from '../../src/components/conversations/ConversationItem';
import type { ConversationSummary } from '../../src/lib/api-client';

const mockConversation: ConversationSummary = {
  id: 'conv-1',
  title: 'Test Conversation',
  system_message: 'You are helpful',
  created_at: '2024-01-01T10:00:00Z',
  updated_at: '2024-01-01T14:30:00Z',
  message_count: 5,
};

const mockUntitledConversation: ConversationSummary = {
  ...mockConversation,
  id: 'conv-2',
  title: null,
};

describe('ConversationItem', () => {
  const defaultProps = {
    conversation: mockConversation,
    isSelected: false,
    onClick: jest.fn(),
    onDelete: jest.fn().mockResolvedValue(undefined),
    onUpdateTitle: jest.fn().mockResolvedValue(undefined),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('displays conversation title', () => {
      render(<ConversationItem {...defaultProps} />);

      expect(screen.getByText('Test Conversation')).toBeInTheDocument();
    });

    it('displays "Untitled" when no title', () => {
      render(<ConversationItem {...defaultProps} conversation={mockUntitledConversation} />);

      expect(screen.getByText('Untitled')).toBeInTheDocument();
    });

    it('displays message count', () => {
      render(<ConversationItem {...defaultProps} />);

      expect(screen.getByText(/5 messages/)).toBeInTheDocument();
    });

    it('displays formatted timestamp with time', () => {
      render(<ConversationItem {...defaultProps} />);

      // Should show date and time (format depends on locale)
      const timestampElement = screen.getByText(/Jan/);
      expect(timestampElement).toBeInTheDocument();
    });

    it('applies selected styling when selected', () => {
      const { container } = render(<ConversationItem {...defaultProps} isSelected={true} />);

      const itemDiv = container.querySelector('.bg-primary\\/10');
      expect(itemDiv).toBeInTheDocument();
    });

    it('applies hover styling when not selected', () => {
      const { container } = render(<ConversationItem {...defaultProps} isSelected={false} />);

      const itemDiv = container.querySelector('.hover\\:bg-muted\\/50');
      expect(itemDiv).toBeInTheDocument();
    });
  });

  describe('clicking', () => {
    it('calls onClick when clicked', () => {
      render(<ConversationItem {...defaultProps} />);

      const item = screen.getByText('Test Conversation');
      fireEvent.click(item);

      expect(defaultProps.onClick).toHaveBeenCalledTimes(1);
    });

    it('does not call onClick when in editing mode', () => {
      render(<ConversationItem {...defaultProps} />);

      // Enter edit mode
      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      // Click on input
      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.click(input);

      // onClick should not be called
      expect(defaultProps.onClick).not.toHaveBeenCalled();
    });
  });

  describe('inline editing', () => {
    it('shows edit input when edit button clicked', () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      expect(screen.getByPlaceholderText('Conversation title')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Test Conversation')).toBeInTheDocument();
    });

    it('allows changing title', () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: 'New Title' } });

      expect(screen.getByDisplayValue('New Title')).toBeInTheDocument();
    });

    it('saves on save button click', async () => {
      render(<ConversationItem {...defaultProps} />);

      // Enter edit mode
      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      // Change title
      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: 'Updated Title' } });

      // Save
      const saveButton = screen.getByTitle('Save');
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(defaultProps.onUpdateTitle).toHaveBeenCalledWith('conv-1', 'Updated Title');
      });
    });

    it('saves on Enter key', async () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: 'New Title' } });
      fireEvent.keyDown(input, { key: 'Enter' });

      await waitFor(() => {
        expect(defaultProps.onUpdateTitle).toHaveBeenCalledWith('conv-1', 'New Title');
      });
    });

    it('saves null when title is empty or whitespace', async () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: '   ' } });

      const saveButton = screen.getByTitle('Save');
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(defaultProps.onUpdateTitle).toHaveBeenCalledWith('conv-1', null);
      });
    });

    it('cancels editing on cancel button click', () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: 'Changed' } });

      const cancelButton = screen.getByTitle('Cancel');
      fireEvent.click(cancelButton);

      // Should revert to original title
      expect(screen.getByText('Test Conversation')).toBeInTheDocument();
      expect(screen.queryByPlaceholderText('Conversation title')).not.toBeInTheDocument();
    });

    it('cancels editing on Escape key', () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: 'Changed' } });
      fireEvent.keyDown(input, { key: 'Escape' });

      // Should revert to original title
      expect(screen.getByText('Test Conversation')).toBeInTheDocument();
      expect(screen.queryByPlaceholderText('Conversation title')).not.toBeInTheDocument();
    });

    it('exits edit mode after successful save', async () => {
      render(<ConversationItem {...defaultProps} />);

      const editButton = screen.getByTitle('Edit title');
      fireEvent.click(editButton);

      const input = screen.getByPlaceholderText('Conversation title');
      fireEvent.change(input, { target: { value: 'New Title' } });

      const saveButton = screen.getByTitle('Save');
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(screen.queryByPlaceholderText('Conversation title')).not.toBeInTheDocument();
      });
    });
  });

  describe('delete', () => {
    it('calls onDelete immediately without confirmation', async () => {
      render(<ConversationItem {...defaultProps} />);

      const deleteButton = screen.getByTitle('Delete permanently');
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(defaultProps.onDelete).toHaveBeenCalledWith('conv-1');
      });
    });

    it('shows deleting state during delete', () => {
      // Mock slow delete
      const slowDelete = jest.fn().mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100)),
      );

      const { container } = render(<ConversationItem {...defaultProps} onDelete={slowDelete} />);

      const deleteButton = screen.getByTitle('Delete permanently');
      fireEvent.click(deleteButton);

      // Should show opacity and disable pointer events
      const itemDiv = container.querySelector('.opacity-50.pointer-events-none');
      expect(itemDiv).toBeInTheDocument();
    });
  });

  describe('action button visibility', () => {
    it('hides action buttons by default (shown on hover via CSS)', () => {
      const { container } = render(<ConversationItem {...defaultProps} />);

      const actionsDiv = container.querySelector('.opacity-0.group-hover\\:opacity-100');
      expect(actionsDiv).toBeInTheDocument();
    });
  });
});

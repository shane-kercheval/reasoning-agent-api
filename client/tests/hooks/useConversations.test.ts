/**
 * Tests for useConversations hook.
 *
 * Tests conversation management with optimistic updates and error handling.
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useConversations } from '../../src/hooks/useConversations';
import { useConversationsStore } from '../../src/store/conversations-store';
import { useToastStore } from '../../src/store/toast-store';
import type { APIClient, ConversationSummary } from '../../src/lib/api-client';

// Mock data
const mockConversation1: ConversationSummary = {
  id: 'conv-1',
  title: 'Test Conversation',
  system_message: 'You are helpful',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  archived_at: null,
  message_count: 3,
};

const mockConversation2: ConversationSummary = {
  id: 'conv-2',
  title: 'Another Conversation',
  system_message: 'You are helpful',
  created_at: '2024-01-02T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
  archived_at: null,
  message_count: 5,
};

// Mock API client
const createMockClient = (): jest.Mocked<APIClient> => ({
  listConversations: jest.fn(),
  getConversation: jest.fn(),
  deleteConversation: jest.fn(),
  archiveConversation: jest.fn(),
  permanentlyDeleteConversation: jest.fn(),
  updateConversationTitle: jest.fn(),
} as any);

describe('useConversations', () => {
  let mockClient: jest.Mocked<APIClient>;

  beforeEach(() => {
    mockClient = createMockClient();

    // Reset stores
    const conversationsStore = useConversationsStore.getState();
    conversationsStore.setConversations([]);
    conversationsStore.setSelectedConversation(null);
    conversationsStore.setLoading(false);
    conversationsStore.clearError();

    const toastStore = useToastStore.getState();
    toastStore.clearAll();

    // Clear all mocks
    jest.clearAllMocks();
  });

  describe('fetchConversations', () => {
    it('fetches and sets conversations on success', async () => {
      mockClient.listConversations.mockResolvedValue({
        conversations: [mockConversation1, mockConversation2],
        total: 2,
      });

      const { result } = renderHook(() => useConversations(mockClient));

      await waitFor(() => {
        expect(result.current.conversations).toHaveLength(2);
      });

      expect(mockClient.listConversations).toHaveBeenCalled();
      expect(result.current.conversations).toEqual([mockConversation1, mockConversation2]);
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('sets loading state during fetch', async () => {
      mockClient.listConversations.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ conversations: [], total: 0 }), 100)),
      );

      const { result } = renderHook(() => useConversations(mockClient));

      // Should be loading initially
      await waitFor(() => {
        expect(result.current.isLoading).toBe(true);
      });
    });

    it('clears loading state after fetch completes', async () => {
      mockClient.listConversations.mockResolvedValue({
        conversations: [mockConversation1],
        total: 1,
      });

      const { result } = renderHook(() => useConversations(mockClient));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('sets loading to false even when API fails', async () => {
      mockClient.listConversations.mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useConversations(mockClient));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('sets error and shows toast on failure', async () => {
      mockClient.listConversations.mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useConversations(mockClient));

      await waitFor(() => {
        expect(result.current.error).toBe('Network error');
      });

      const toasts = useToastStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].type).toBe('error');
      expect(toasts[0].message).toBe('Network error');
    });
  });

  describe('deleteConversation (optimistic update)', () => {
    it('optimistically removes conversation and calls API', async () => {
      mockClient.permanentlyDeleteConversation.mockResolvedValue();

      // Set up initial state
      const conversationsStore = useConversationsStore.getState();
      conversationsStore.setConversations([mockConversation1, mockConversation2]);

      const { result } = renderHook(() => useConversations(mockClient));

      await act(async () => {
        await result.current.deleteConversation('conv-1');
      });

      expect(mockClient.permanentlyDeleteConversation).toHaveBeenCalledWith('conv-1');
      expect(result.current.conversations).toHaveLength(1);
      expect(result.current.conversations[0].id).toBe('conv-2');

      // Should show success toast
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.type === 'success')).toBe(true);
    });

    it('rollsback on API failure', async () => {
      mockClient.permanentlyDeleteConversation.mockRejectedValue(new Error('Failed to delete conversation'));

      // Set up initial state
      const conversationsStore = useConversationsStore.getState();
      conversationsStore.setConversations([mockConversation1, mockConversation2]);

      const { result } = renderHook(() => useConversations(mockClient));

      await act(async () => {
        try {
          await result.current.deleteConversation('conv-1');
        } catch (err) {
          // Expected to throw
        }
      });

      // Should restore conversation
      expect(result.current.conversations).toHaveLength(2);
      expect(result.current.conversations.some((c) => c.id === 'conv-1')).toBe(true);

      // Should show error toast
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.type === 'error' && t.message === 'Failed to delete conversation')).toBe(true);
    });

    it('shows error toast when conversation not found', async () => {
      const conversationsStore = useConversationsStore.getState();
      conversationsStore.setConversations([mockConversation1]);

      const { result } = renderHook(() => useConversations(mockClient));

      await act(async () => {
        await result.current.deleteConversation('non-existent');
      });

      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.message === 'Conversation not found')).toBe(true);
      expect(mockClient.deleteConversation).not.toHaveBeenCalled();
    });
  });

  describe('updateConversationTitle (optimistic update)', () => {
    it('optimistically updates title and calls API', async () => {
      const updatedConv: ConversationSummary = {
        ...mockConversation1,
        title: 'Updated Title',
        updated_at: '2024-01-03T00:00:00Z',
      };
      mockClient.updateConversationTitle.mockResolvedValue(updatedConv);

      // Set up initial state
      const conversationsStore = useConversationsStore.getState();
      conversationsStore.setConversations([mockConversation1, mockConversation2]);

      const { result } = renderHook(() => useConversations(mockClient));

      await act(async () => {
        await result.current.updateConversationTitle('conv-1', 'Updated Title');
      });

      expect(mockClient.updateConversationTitle).toHaveBeenCalledWith('conv-1', 'Updated Title');
      expect(result.current.conversations[0].title).toBe('Updated Title');

      // Should show success toast
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.type === 'success' && t.message === 'Title updated')).toBe(true);
    });

    it('rollsback on API failure', async () => {
      mockClient.updateConversationTitle.mockRejectedValue(new Error('Update failed'));

      // Set up initial state
      const conversationsStore = useConversationsStore.getState();
      conversationsStore.setConversations([mockConversation1, mockConversation2]);

      const { result } = renderHook(() => useConversations(mockClient));

      await act(async () => {
        try {
          await result.current.updateConversationTitle('conv-1', 'New Title');
        } catch (err) {
          // Expected to throw
        }
      });

      // Should restore original title
      expect(result.current.conversations[0].title).toBe(mockConversation1.title);

      // Should show error toast
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.type === 'error' && t.message === 'Update failed')).toBe(true);
    });

    it('handles null title', async () => {
      const updatedConv: ConversationSummary = {
        ...mockConversation1,
        title: null,
        updated_at: '2024-01-03T00:00:00Z',
      };
      mockClient.updateConversationTitle.mockResolvedValue(updatedConv);

      const conversationsStore = useConversationsStore.getState();
      conversationsStore.setConversations([mockConversation1]);

      const { result } = renderHook(() => useConversations(mockClient));

      await act(async () => {
        await result.current.updateConversationTitle('conv-1', null);
      });

      expect(mockClient.updateConversationTitle).toHaveBeenCalledWith('conv-1', null);
      expect(result.current.conversations[0].title).toBeNull();
    });
  });

  describe('selectConversation', () => {
    it('sets selected conversation id', () => {
      const { result } = renderHook(() => useConversations(mockClient));

      act(() => {
        result.current.selectConversation('conv-1');
      });

      expect(result.current.selectedConversationId).toBe('conv-1');
    });

    it('can clear selection', () => {
      const { result } = renderHook(() => useConversations(mockClient));

      act(() => {
        result.current.selectConversation('conv-1');
        result.current.selectConversation(null);
      });

      expect(result.current.selectedConversationId).toBeNull();
    });
  });
});

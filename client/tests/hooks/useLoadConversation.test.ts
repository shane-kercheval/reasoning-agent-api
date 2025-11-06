/**
 * Tests for useLoadConversation hook.
 *
 * Tests loading conversation history and message conversion.
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useLoadConversation } from '../../src/hooks/useLoadConversation';
import { useToastStore } from '../../src/store/toast-store';
import type { APIClient, ConversationDetail, ConversationMessage } from '../../src/lib/api-client';

// Mock conversation data
const mockMessages: ConversationMessage[] = [
  {
    id: 'msg-1',
    conversation_id: 'conv-1',
    role: 'user',
    content: 'Hello',
    reasoning_events: null,
    created_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'msg-2',
    conversation_id: 'conv-1',
    role: 'assistant',
    content: 'Hi there!',
    reasoning_events: [
      {
        type: 'planning',
        step_iteration: 1,
        metadata: {},
      },
    ],
    created_at: '2024-01-01T00:00:01Z',
  },
  {
    id: 'msg-3',
    conversation_id: 'conv-1',
    role: 'system',
    content: 'You are helpful',
    reasoning_events: null,
    created_at: '2024-01-01T00:00:00Z',
  },
];

const mockConversation: ConversationDetail = {
  id: 'conv-1',
  title: 'Test Conversation',
  system_message: 'You are helpful',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:01Z',
  messages: mockMessages,
};

// Mock API client
const createMockClient = (): jest.Mocked<APIClient> => ({
  getConversation: jest.fn(),
} as any);

describe('useLoadConversation', () => {
  let mockClient: jest.Mocked<APIClient>;

  beforeEach(() => {
    mockClient = createMockClient();

    // Reset toast store
    const toastStore = useToastStore.getState();
    toastStore.clearAll();

    // Clear all mocks
    jest.clearAllMocks();
  });

  describe('loadConversation', () => {
    it('loads and converts conversation messages', async () => {
      mockClient.getConversation.mockResolvedValue(mockConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      let displayMessages;
      await act(async () => {
        displayMessages = await result.current.loadConversation('conv-1');
      });

      expect(mockClient.getConversation).toHaveBeenCalledWith('conv-1');
      expect(displayMessages).toHaveLength(3);
      expect(result.current.messages).toHaveLength(3);
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('converts user messages correctly', async () => {
      mockClient.getConversation.mockResolvedValue(mockConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      const userMessage = result.current.messages.find((m) => m.role === 'user');
      expect(userMessage).toBeDefined();
      expect(userMessage?.content).toBe('Hello');
      expect(userMessage?.reasoningEvents).toBeUndefined();
    });

    it('converts assistant messages with reasoning events', async () => {
      mockClient.getConversation.mockResolvedValue(mockConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      const assistantMessage = result.current.messages.find((m) => m.role === 'assistant');
      expect(assistantMessage).toBeDefined();
      expect(assistantMessage?.content).toBe('Hi there!');
      expect(assistantMessage?.reasoningEvents).toBeDefined();
      expect(assistantMessage?.reasoningEvents).toHaveLength(1);
    });

    it('converts system messages correctly', async () => {
      mockClient.getConversation.mockResolvedValue(mockConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      const systemMessage = result.current.messages.find((m) => m.role === 'system');
      expect(systemMessage).toBeDefined();
      expect(systemMessage?.content).toBe('You are helpful');
    });

    it('filters out messages with no content', async () => {
      const conversationWithEmptyMessage: ConversationDetail = {
        ...mockConversation,
        messages: [
          ...mockMessages,
          {
            id: 'msg-4',
            conversation_id: 'conv-1',
            role: 'user',
            content: '',
            reasoning_events: null,
            created_at: '2024-01-01T00:00:02Z',
          },
        ],
      };

      mockClient.getConversation.mockResolvedValue(conversationWithEmptyMessage);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      // Should still be 3 (empty message filtered out)
      expect(result.current.messages).toHaveLength(3);
    });

    it('sets loading state during fetch', async () => {
      mockClient.getConversation.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(mockConversation), 100)),
      );

      const { result } = renderHook(() => useLoadConversation(mockClient));

      act(() => {
        result.current.loadConversation('conv-1');
      });

      // Should be loading
      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('clears loading state after success', async () => {
      mockClient.getConversation.mockResolvedValue(mockConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      expect(result.current.isLoading).toBe(false);
    });

    it('clears loading state even on failure', async () => {
      mockClient.getConversation.mockRejectedValue(new Error('Load failed'));

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        try {
          await result.current.loadConversation('conv-1');
        } catch (err) {
          // Expected to throw
        }
      });

      expect(result.current.isLoading).toBe(false);
    });

    it('sets error and shows toast on failure', async () => {
      mockClient.getConversation.mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        try {
          await result.current.loadConversation('conv-1');
        } catch (err) {
          // Expected to throw
        }
      });

      expect(result.current.error).toBe('Network error');

      const toasts = useToastStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].type).toBe('error');
      expect(toasts[0].message).toBe('Network error');
    });

    it('throws error on failure for error handling in UI', async () => {
      mockClient.getConversation.mockRejectedValue(new Error('API error'));

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await expect(
        act(async () => {
          await result.current.loadConversation('conv-1');
        }),
      ).rejects.toThrow('API error');
    });

    it('handles empty conversation', async () => {
      const emptyConversation: ConversationDetail = {
        ...mockConversation,
        messages: [],
      };

      mockClient.getConversation.mockResolvedValue(emptyConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      expect(result.current.messages).toHaveLength(0);
      expect(result.current.error).toBeNull();
    });
  });

  describe('clearMessages', () => {
    it('clears messages and error', async () => {
      mockClient.getConversation.mockResolvedValue(mockConversation);

      const { result } = renderHook(() => useLoadConversation(mockClient));

      await act(async () => {
        await result.current.loadConversation('conv-1');
      });

      expect(result.current.messages).toHaveLength(3);

      act(() => {
        result.current.clearMessages();
      });

      expect(result.current.messages).toHaveLength(0);
      expect(result.current.error).toBeNull();
    });
  });
});

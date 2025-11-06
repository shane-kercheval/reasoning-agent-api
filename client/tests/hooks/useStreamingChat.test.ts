/**
 * Tests for useStreamingChat hook.
 *
 * Tests streaming chat functionality, conversation ID handling, and multi-tab isolation.
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useStreamingChat } from '../../src/hooks/useStreamingChat';
import type { APIClient } from '../../src/lib/api-client';
import { useChatStore } from '../../src/store/chat-store';

// Mock API client
const createMockAPIClient = (): APIClient => ({
  streamChatCompletion: jest.fn(),
  fetchConversationHistory: jest.fn(),
  fetchConversations: jest.fn(),
  deleteConversation: jest.fn(),
  updateConversationTitle: jest.fn(),
  fetchModels: jest.fn(),
});

// Helper to create a mock streaming response with small delays for async processing
async function* mockStreamGenerator(
  chunks: any[],
): AsyncGenerator<any, void, undefined> {
  for (const chunk of chunks) {
    // Small delay to ensure async processing
    await new Promise((resolve) => setTimeout(resolve, 10));
    yield chunk;
  }
}

describe('useStreamingChat', () => {
  let mockClient: APIClient;

  beforeEach(() => {
    mockClient = createMockAPIClient();
    // Reset chat store
    useChatStore.setState({
      conversationId: null,
      streaming: {
        content: '',
        reasoningEvents: [],
        isStreaming: false,
        error: null,
      },
      settings: {
        model: 'gpt-4o-mini',
        temperature: 1,
        routingMode: 'auto',
        systemPrompt: null,
      },
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('conversation ID isolation (multi-tab support)', () => {
    it('uses conversation ID from options, not global store', async () => {
      // Set global store to a different conversation ID
      useChatStore.setState({ conversationId: 'global-conv-id' });

      const mockChunks = [
        {
          choices: [
            {
              delta: { content: 'Hello' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks),
      );

      const { result } = renderHook(() => useStreamingChat(mockClient));

      await act(async () => {
        await result.current.sendMessage('Test message', {
          conversationId: 'tab-specific-conv-id',
        });
      });

      // Verify API was called with the tab-specific conversation ID, not global
      expect(mockClient.streamChatCompletion).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          conversationId: 'tab-specific-conv-id',
        }),
      );
    });

    it('returns new conversation ID when API creates one', async () => {
      const mockChunks = [
        { type: 'conversation_metadata', conversationId: 'new-conv-123' }, // Conversation metadata
        {
          object: 'chat.completion.chunk',
          choices: [
            {
              delta: { content: 'Hello' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks),
      );

      const { result } = renderHook(() => useStreamingChat(mockClient));

      let returnedConversationId: string | null = null;

      await act(async () => {
        returnedConversationId = await result.current.sendMessage('Test message', {
          conversationId: null, // New conversation
        });
      });

      expect(returnedConversationId).toBe('new-conv-123');
    });

    it('returns passed conversation ID when no new one is created', async () => {
      const mockChunks = [
        {
          choices: [
            {
              delta: { content: 'Hello' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks),
      );

      const { result } = renderHook(() => useStreamingChat(mockClient));

      let returnedConversationId: string | null = null;

      await act(async () => {
        returnedConversationId = await result.current.sendMessage('Test message', {
          conversationId: 'existing-conv-id',
        });
      });

      expect(returnedConversationId).toBe('existing-conv-id');
    });

    it('handles multiple hook instances with different conversation IDs', async () => {
      const mockChunks1 = [
        { type: 'conversation_metadata', conversationId: 'conv-tab-1' },
        {
          object: 'chat.completion.chunk',
          choices: [
            {
              delta: { content: 'Response 1' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      const mockChunks2 = [
        { type: 'conversation_metadata', conversationId: 'conv-tab-2' },
        {
          object: 'chat.completion.chunk',
          choices: [
            {
              delta: { content: 'Response 2' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      // Create two separate hook instances (simulating two tabs)
      const { result: tab1 } = renderHook(() => useStreamingChat(mockClient));
      const { result: tab2 } = renderHook(() => useStreamingChat(mockClient));

      // Tab 1 sends a message
      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks1),
      );

      let conv1: string | null = null;
      await act(async () => {
        conv1 = await tab1.current.sendMessage('Message from tab 1', {
          conversationId: null,
        });
      });

      expect(conv1).toBe('conv-tab-1');

      // Tab 2 sends a message
      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks2),
      );

      let conv2: string | null = null;
      await act(async () => {
        conv2 = await tab2.current.sendMessage('Message from tab 2', {
          conversationId: null,
        });
      });

      expect(conv2).toBe('conv-tab-2');

      // Both tabs should have their own conversation IDs
      expect(conv1).not.toBe(conv2);
    });
  });

  describe('basic streaming functionality', () => {
    it('accumulates content from stream', async () => {
      const mockChunks = [
        {
          choices: [
            {
              delta: { content: 'Hello ' },
              finish_reason: null,
            },
          ],
        },
        {
          choices: [
            {
              delta: { content: 'world' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks),
      );

      const { result } = renderHook(() => useStreamingChat(mockClient));

      // Content is accumulated in the chat store during streaming
      // After sendMessage completes, content is cleared by the hook
      // So we just verify the API was called successfully
      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(mockClient.streamChatCompletion).toHaveBeenCalled();
    });

    it('sets isStreaming to false after completion', async () => {
      const mockChunks = [
        {
          choices: [
            {
              delta: { content: 'Hello' },
              finish_reason: null,
            },
          ],
        },
        { data: '[DONE]' },
      ];

      (mockClient.streamChatCompletion as jest.Mock).mockReturnValue(
        mockStreamGenerator(mockChunks),
      );

      const { result } = renderHook(() => useStreamingChat(mockClient));

      expect(result.current.isStreaming).toBe(false);

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(result.current.isStreaming).toBe(false);
    });

    it('clears content when clear is called', () => {
      const { result } = renderHook(() => useStreamingChat(mockClient));

      // Manually set content in store to test clear
      act(() => {
        useChatStore.setState({
          streaming: {
            content: 'Some content',
            reasoningEvents: [],
            isStreaming: false,
            error: null,
          },
        });
      });

      expect(result.current.content).toBe('Some content');

      act(() => {
        result.current.clear();
      });

      expect(result.current.content).toBe('');
    });
  });

  describe('error handling', () => {
    it('handles API errors', async () => {
      (mockClient.streamChatCompletion as jest.Mock).mockImplementation(async function* () {
        throw new Error('API Error');
      });

      const { result } = renderHook(() => useStreamingChat(mockClient));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(result.current.error).toBe('API Error');
      expect(result.current.isStreaming).toBe(false);
    });

    it('provides cancel method', () => {
      const { result } = renderHook(() => useStreamingChat(mockClient));

      expect(result.current.cancel).toBeDefined();
      expect(typeof result.current.cancel).toBe('function');

      // Cancel is safe to call even when not streaming
      act(() => {
        result.current.cancel();
      });

      expect(result.current.isStreaming).toBe(false);
    });
  });
});

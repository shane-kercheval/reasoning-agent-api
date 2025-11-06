/**
 * React hook for loading conversation history.
 *
 * Fetches conversation messages from backend and returns them formatted
 * for display in the chat interface.
 */

import { useState, useCallback } from 'react';
import type { APIClient, ConversationDetail, ConversationMessage } from '../lib/api-client';
import type { ReasoningEvent } from '../types/openai';

/**
 * Message formatted for chat display.
 */
export interface DisplayMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
}

/**
 * Hook for loading conversation history.
 *
 * @param apiClient - Configured API client
 * @returns State and actions for loading conversations
 *
 * @example
 * ```typescript
 * const { client } = useAPIClient();
 * const { messages, isLoading, error, loadConversation } = useLoadConversation(client);
 *
 * // Load a conversation
 * await loadConversation('conversation-uuid');
 * ```
 */
export function useLoadConversation(apiClient: APIClient) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Convert API message to display format.
   */
  const convertMessage = useCallback((msg: ConversationMessage): DisplayMessage | null => {
    // Skip messages with no content
    if (!msg.content) {
      return null;
    }

    return {
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content,
      reasoningEvents: msg.reasoning_events
        ? (msg.reasoning_events as unknown as ReasoningEvent[])
        : undefined,
    };
  }, []);

  /**
   * Load conversation history from API.
   */
  const loadConversation = useCallback(
    async (conversationId: string): Promise<DisplayMessage[]> => {
      setIsLoading(true);
      setError(null);

      try {
        const conversation: ConversationDetail = await apiClient.getConversation(conversationId);

        // Convert messages to display format
        const displayMessages: DisplayMessage[] = conversation.messages
          .map(convertMessage)
          .filter((msg): msg is DisplayMessage => msg !== null);

        setMessages(displayMessages);
        return displayMessages;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load conversation';
        setError(errorMessage);
        throw new Error(errorMessage);
      } finally {
        setIsLoading(false);
      }
    },
    [apiClient, convertMessage],
  );

  /**
   * Clear loaded messages.
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    loadConversation,
    clearMessages,
  };
}

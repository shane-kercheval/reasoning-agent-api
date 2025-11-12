/**
 * React hook for loading conversation history.
 *
 * Fetches conversation messages from backend and returns them formatted
 * for display in the chat interface.
 */

import { useState, useCallback } from 'react';
import type { APIClient, ConversationDetail, ConversationMessage } from '../lib/api-client';
import type { ReasoningEvent, Usage } from '../types/openai';
import { useToast } from '../store/toast-store';

/**
 * Message formatted for chat display.
 */
export interface DisplayMessage {
  id?: string;  // UUID from database
  sequenceNumber?: number;  // Sequence number from database
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
  usage?: Usage | null;
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
  const toast = useToast();

  /**
   * Convert API message to display format.
   * Extracts usage data from metadata for display.
   */
  const convertMessage = useCallback((msg: ConversationMessage): DisplayMessage | null => {
    if (!msg.content) {
      return null;
    }

    let usage: Usage | undefined = undefined;
    if (msg.metadata?.usage) {
      usage = {
        ...(msg.metadata.usage as Usage),
        ...(msg.metadata.cost && {
          prompt_cost: msg.metadata.cost.prompt_cost,
          completion_cost: msg.metadata.cost.completion_cost,
          total_cost: msg.metadata.cost.total_cost,
        }),
      };
    }

    return {
      id: msg.id,
      sequenceNumber: msg.sequence_number,
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content,
      reasoningEvents: msg.reasoning_events
        ? (msg.reasoning_events as unknown as ReasoningEvent[])
        : undefined,
      usage,
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
        toast.error(errorMessage);
        throw new Error(errorMessage);
      } finally {
        setIsLoading(false);
      }
    },
    [apiClient, convertMessage, toast],
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

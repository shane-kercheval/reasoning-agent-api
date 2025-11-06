/**
 * React hook for managing conversations.
 *
 * Provides actions for fetching, deleting, and updating conversations.
 * Uses Zustand store for state management and API client for backend calls.
 */

import { useCallback, useEffect } from 'react';
import type { APIClient } from '../lib/api-client';
import { useConversationsStore } from '../store/conversations-store';

/**
 * Hook for conversation management.
 *
 * @param apiClient - Configured API client
 * @returns Conversation state and actions
 *
 * @example
 * ```typescript
 * const { client } = useAPIClient();
 * const {
 *   conversations,
 *   isLoading,
 *   error,
 *   fetchConversations,
 *   deleteConversation,
 *   updateConversationTitle
 * } = useConversations(client);
 * ```
 */
export function useConversations(apiClient: APIClient) {
  // Get state and actions from store
  const {
    conversations,
    isLoading,
    error,
    selectedConversationId,
    setConversations,
    updateConversation,
    removeConversation,
    setSelectedConversation,
    setLoading,
    setError,
    clearError,
  } = useConversationsStore();

  /**
   * Fetch conversations from API.
   */
  const fetchConversations = useCallback(
    async (options?: { limit?: number; offset?: number }) => {
      setLoading(true);
      clearError();

      try {
        const response = await apiClient.listConversations(options);
        setConversations(response.conversations);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch conversations');
      } finally {
        setLoading(false);
      }
    },
    [apiClient, setLoading, clearError, setConversations, setError],
  );

  /**
   * Delete a conversation.
   */
  const deleteConversation = useCallback(
    async (conversationId: string) => {
      try {
        await apiClient.deleteConversation(conversationId);
        removeConversation(conversationId);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete conversation');
        throw err; // Re-throw so UI can handle it
      }
    },
    [apiClient, removeConversation, setError],
  );

  /**
   * Update conversation title.
   */
  const updateConversationTitle = useCallback(
    async (conversationId: string, title: string | null) => {
      try {
        const updated = await apiClient.updateConversationTitle(conversationId, title);
        updateConversation(conversationId, { title: updated.title, updated_at: updated.updated_at });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to update conversation title');
        throw err;
      }
    },
    [apiClient, updateConversation, setError],
  );

  /**
   * Select a conversation (for loading it).
   */
  const selectConversation = useCallback(
    (conversationId: string | null) => {
      setSelectedConversation(conversationId);
    },
    [setSelectedConversation],
  );

  // Fetch conversations on mount
  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  return {
    // State
    conversations,
    isLoading,
    error,
    selectedConversationId,

    // Actions
    fetchConversations,
    deleteConversation,
    updateConversationTitle,
    selectConversation,
  };
}

/**
 * React hook for managing conversations.
 *
 * Provides actions for fetching, deleting, and updating conversations.
 * Uses Zustand store for state management and API client for backend calls.
 * Implements optimistic updates with rollback on error.
 */

import { useCallback, useEffect } from 'react';
import type { APIClient } from '../lib/api-client';
import { useConversationsStore } from '../store/conversations-store';
import { useToast } from '../store/toast-store';

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
    restoreConversation,
    setSelectedConversation,
    setLoading,
    setError,
    clearError,
  } = useConversationsStore();

  // Toast notifications
  const toast = useToast();

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
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch conversations';
        setError(errorMessage);
        toast.error(errorMessage);
      } finally {
        setLoading(false);
      }
    },
    [apiClient, setLoading, clearError, setConversations, setError, toast],
  );

  /**
   * Delete a conversation permanently (optimistic update).
   * Uses optimistic update for immediate feedback since deletion is final.
   */
  const deleteConversation = useCallback(
    async (conversationId: string) => {
      // Find conversation for potential rollback
      const conversation = conversations.find((c) => c.id === conversationId);
      if (!conversation) {
        toast.error('Conversation not found');
        return;
      }

      // Optimistically remove from UI (immediate feedback for destructive action)
      removeConversation(conversationId);

      try {
        await apiClient.permanentlyDeleteConversation(conversationId);
        toast.success('Conversation permanently deleted');
      } catch (err) {
        // Rollback: restore the conversation
        restoreConversation(conversation);
        const errorMessage = err instanceof Error ? err.message : 'Failed to delete conversation';
        setError(errorMessage);
        toast.error(errorMessage);
        throw err; // Re-throw so UI can handle it
      }
    },
    [apiClient, conversations, removeConversation, restoreConversation, setError, toast],
  );

  /**
   * Archive a conversation (soft delete).
   * Refreshes conversation list after archiving to ensure correct state.
   */
  const archiveConversation = useCallback(
    async (conversationId: string) => {
      // Verify conversation exists
      const conversation = conversations.find((c) => c.id === conversationId);
      if (!conversation) {
        toast.error('Conversation not found');
        return;
      }

      try {
        await apiClient.archiveConversation(conversationId);
        toast.success('Conversation archived');
        // Refresh to get updated archived_at timestamp
        await fetchConversations();
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to archive conversation';
        setError(errorMessage);
        toast.error(errorMessage);
        throw err; // Re-throw so UI can handle it
      }
    },
    [apiClient, conversations, fetchConversations, setError, toast],
  );

  /**
   * Update conversation title (optimistic update).
   */
  const updateConversationTitle = useCallback(
    async (conversationId: string, title: string | null) => {
      // Find conversation for potential rollback
      const conversation = conversations.find((c) => c.id === conversationId);
      if (!conversation) {
        toast.error('Conversation not found');
        return;
      }

      // Save original title for rollback
      const originalTitle = conversation.title;
      const originalUpdatedAt = conversation.updated_at;

      // Optimistically update UI
      updateConversation(conversationId, { title, updated_at: new Date().toISOString() });

      try {
        const updated = await apiClient.updateConversationTitle(conversationId, title);
        // Update with actual server response
        updateConversation(conversationId, { title: updated.title, updated_at: updated.updated_at });
        toast.success('Title updated');
      } catch (err) {
        // Rollback: restore original title
        updateConversation(conversationId, { title: originalTitle, updated_at: originalUpdatedAt });
        const errorMessage = err instanceof Error ? err.message : 'Failed to update conversation title';
        setError(errorMessage);
        toast.error(errorMessage);
        throw err;
      }
    },
    [apiClient, conversations, updateConversation, setError, toast],
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount

  return {
    // State
    conversations,
    isLoading,
    error,
    selectedConversationId,

    // Actions
    fetchConversations,
    deleteConversation,
    archiveConversation,
    updateConversationTitle,
    selectConversation,
  };
}

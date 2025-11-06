/**
 * Zustand store for conversation list management.
 *
 * Manages the conversation sidebar state separately from chat state.
 * Handles fetching, deleting, and updating conversations.
 */

import { create } from 'zustand';
import type { ConversationSummary } from '../lib/api-client';

// ============================================================================
// Types
// ============================================================================

interface ConversationsStore {
  // State
  conversations: ConversationSummary[];
  isLoading: boolean;
  error: string | null;
  selectedConversationId: string | null;

  // Actions
  setConversations: (conversations: ConversationSummary[]) => void;
  addConversation: (conversation: ConversationSummary) => void;
  updateConversation: (id: string, updates: Partial<ConversationSummary>) => void;
  removeConversation: (id: string) => void;
  setSelectedConversation: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

// ============================================================================
// Store
// ============================================================================

export const useConversationsStore = create<ConversationsStore>((set) => ({
  // Initial state
  conversations: [],
  isLoading: false,
  error: null,
  selectedConversationId: null,

  // Actions
  setConversations: (conversations) => set({ conversations, error: null }),

  addConversation: (conversation) =>
    set((state) => ({
      conversations: [conversation, ...state.conversations],
    })),

  updateConversation: (id, updates) =>
    set((state) => ({
      conversations: state.conversations.map((conv) =>
        conv.id === id ? { ...conv, ...updates } : conv,
      ),
    })),

  removeConversation: (id) =>
    set((state) => ({
      conversations: state.conversations.filter((conv) => conv.id !== id),
      selectedConversationId: state.selectedConversationId === id ? null : state.selectedConversationId,
    })),

  setSelectedConversation: (id) => set({ selectedConversationId: id }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error, isLoading: false }),

  clearError: () => set({ error: null }),
}));

// ============================================================================
// Selectors (for optimized subscriptions)
// ============================================================================

export const useConversations = () => useConversationsStore((state) => state.conversations);
export const useConversationsLoading = () => useConversationsStore((state) => state.isLoading);
export const useConversationsError = () => useConversationsStore((state) => state.error);
export const useSelectedConversation = () =>
  useConversationsStore((state) => state.selectedConversationId);

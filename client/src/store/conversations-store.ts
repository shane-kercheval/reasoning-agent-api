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

export type ConversationViewFilter = 'active' | 'archived';

interface ConversationsStore {
  // State
  conversations: ConversationSummary[];
  isLoading: boolean;
  error: string | null;
  selectedConversationId: string | null;
  viewFilter: ConversationViewFilter;
  searchQuery: string;

  // Actions
  setConversations: (conversations: ConversationSummary[]) => void;
  addConversation: (conversation: ConversationSummary) => void;
  updateConversation: (id: string, updates: Partial<ConversationSummary>) => void;
  removeConversation: (id: string) => void;
  restoreConversation: (conversation: ConversationSummary) => void;
  setSelectedConversation: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  setViewFilter: (filter: ConversationViewFilter) => void;
  setSearchQuery: (query: string) => void;
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
  viewFilter: 'active',
  searchQuery: '',

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

  restoreConversation: (conversation) =>
    set((state) => {
      // Check if conversation already exists
      const exists = state.conversations.some((c) => c.id === conversation.id);
      if (exists) {
        // Update existing conversation
        return {
          conversations: state.conversations.map((c) =>
            c.id === conversation.id ? conversation : c,
          ),
        };
      }
      // Add back to list (restore deleted conversation)
      return {
        conversations: [conversation, ...state.conversations],
      };
    }),

  setSelectedConversation: (id) => set({ selectedConversationId: id }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error, isLoading: false }),

  clearError: () => set({ error: null }),

  setViewFilter: (filter) => set({ viewFilter: filter }),

  setSearchQuery: (query) => set({ searchQuery: query }),
}));

// ============================================================================
// Selectors (for optimized subscriptions)
// ============================================================================

export const useConversations = () => useConversationsStore((state) => state.conversations);
export const useConversationsLoading = () => useConversationsStore((state) => state.isLoading);
export const useConversationsError = () => useConversationsStore((state) => state.error);
export const useSelectedConversation = () =>
  useConversationsStore((state) => state.selectedConversationId);
export const useViewFilter = () => useConversationsStore((state) => state.viewFilter);
export const useSearchQuery = () => useConversationsStore((state) => state.searchQuery);

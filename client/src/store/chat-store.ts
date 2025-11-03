/**
 * Zustand store for chat application state.
 *
 * Architecture:
 * - conversation_id: Tracked here, messages stored in backend PostgreSQL
 * - settings: Persisted to localStorage via persist middleware
 * - streaming state: In-memory only (content, reasoning events, errors)
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ReasoningEvent } from '../types/openai';
import type { RoutingModeType } from '../constants';
import { RoutingMode, APIDefaults } from '../constants';

// ============================================================================
// Types
// ============================================================================

export interface ChatSettings {
  model: string;
  routingMode: RoutingModeType;
  temperature: number;
  systemPrompt: string;
}

interface StreamingState {
  isStreaming: boolean;
  content: string;
  reasoningEvents: ReasoningEvent[];
  error: string | null;
}

interface ChatStore {
  // Conversation tracking (not persisted - backend manages history)
  conversationId: string | null;

  // User settings (persisted to localStorage)
  settings: ChatSettings;

  // Streaming state (not persisted - ephemeral)
  streaming: StreamingState;

  // Actions
  setConversationId: (id: string | null) => void;
  updateSettings: (settings: Partial<ChatSettings>) => void;
  startStreaming: () => void;
  appendContent: (content: string) => void;
  addReasoningEvent: (event: ReasoningEvent) => void;
  setError: (error: string | null) => void;
  stopStreaming: () => void;
  clearStreaming: () => void;
  newConversation: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialStreamingState: StreamingState = {
  isStreaming: false,
  content: '',
  reasoningEvents: [],
  error: null,
};

const initialSettings: ChatSettings = {
  model: APIDefaults.MODEL,
  routingMode: RoutingMode.REASONING as RoutingModeType,
  temperature: APIDefaults.TEMPERATURE,
  systemPrompt: '',
};

// ============================================================================
// Store
// ============================================================================

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      // State
      conversationId: null,
      settings: initialSettings,
      streaming: initialStreamingState,

      // Actions
      setConversationId: (id) => set({ conversationId: id }),

      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),

      startStreaming: () =>
        set((state) => ({
          streaming: {
            ...state.streaming,
            isStreaming: true,
            error: null,
          },
        })),

      appendContent: (content) =>
        set((state) => ({
          streaming: {
            ...state.streaming,
            content: state.streaming.content + content,
          },
        })),

      addReasoningEvent: (event) =>
        set((state) => ({
          streaming: {
            ...state.streaming,
            reasoningEvents: [...state.streaming.reasoningEvents, event],
          },
        })),

      setError: (error) =>
        set((state) => ({
          streaming: {
            ...state.streaming,
            error,
            isStreaming: false,
          },
        })),

      stopStreaming: () =>
        set((state) => ({
          streaming: {
            ...state.streaming,
            isStreaming: false,
          },
        })),

      clearStreaming: () =>
        set({
          streaming: initialStreamingState,
        }),

      newConversation: () =>
        set({
          conversationId: null,
          streaming: initialStreamingState,
        }),
    }),
    {
      name: 'reasoning-agent-settings', // localStorage key
      storage: createJSONStorage(() => localStorage),
      // Only persist settings, not conversationId or streaming state
      partialize: (state) => ({ settings: state.settings }),
    },
  ),
);

// ============================================================================
// Selectors (for optimized subscriptions)
// ============================================================================

export const useSettings = () => useChatStore((state) => state.settings);
export const useConversationId = () => useChatStore((state) => state.conversationId);
export const useStreaming = () => useChatStore((state) => state.streaming);
export const useIsStreaming = () => useChatStore((state) => state.streaming.isStreaming);

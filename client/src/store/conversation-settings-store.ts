/**
 * Zustand store for per-conversation settings.
 *
 * Each conversation remembers its own settings (model, temperature, routing mode, system prompt).
 * New conversations start with defaults. Loading a conversation restores its settings.
 * Uses FIFO eviction with max 1000 conversations to prevent unbounded growth.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatSettings } from './chat-store';
import { APIDefaults, RoutingMode } from '../constants';

const MAX_CONVERSATION_SETTINGS = 1000;

interface ConversationSettingsEntry {
  settings: ChatSettings;
  timestamp: number;
}

interface ConversationSettingsStore {
  conversationSettings: Record<string, ConversationSettingsEntry>;

  getSettings: (conversationId: string | null) => ChatSettings;
  saveSettings: (conversationId: string, settings: ChatSettings) => void;
  clearSettings: (conversationId: string) => void;
  clearAll: () => void;
}

const defaultSettings: ChatSettings = {
  model: APIDefaults.MODEL,
  routingMode: RoutingMode.AUTO,
  temperature: APIDefaults.TEMPERATURE,
  systemPrompt: '',
  reasoningEffort: undefined,
  contextUtilization: 'full',
};

export const useConversationSettingsStore = create<ConversationSettingsStore>()(
  persist(
    (set, get) => ({
      conversationSettings: {},

      getSettings: (conversationId) => {
        if (!conversationId) {
          return { ...defaultSettings };
        }

        const entry = get().conversationSettings[conversationId];
        if (!entry) {
          return { ...defaultSettings };
        }

        // Merge with defaults to ensure new fields are present
        return { ...defaultSettings, ...entry.settings };
      },

      saveSettings: (conversationId, settings) => {
        set((state) => {
          const newSettings = { ...state.conversationSettings };

          if (Object.keys(newSettings).length >= MAX_CONVERSATION_SETTINGS) {
            const sortedEntries = Object.entries(newSettings).sort(
              ([, a], [, b]) => a.timestamp - b.timestamp
            );
            const oldestKey = sortedEntries[0][0];
            delete newSettings[oldestKey];
          }

          newSettings[conversationId] = {
            settings,
            timestamp: Date.now(),
          };

          return { conversationSettings: newSettings };
        });
      },

      clearSettings: (conversationId) => {
        set((state) => {
          const newSettings = { ...state.conversationSettings };
          delete newSettings[conversationId];
          return { conversationSettings: newSettings };
        });
      },

      clearAll: () => {
        set({ conversationSettings: {} });
      },
    }),
    {
      name: 'conversation-settings-storage',
    }
  )
);

/**
 * Zustand store for tab management.
 *
 * Manages multiple chat tabs, allowing users to have multiple conversations
 * open simultaneously.
 */

import { create } from 'zustand';
import type { ReasoningEvent, Usage } from '../types/openai';
import type { ChatSettings } from './chat-store';

// ============================================================================
// Types
// ============================================================================

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
  usage?: Usage | null;
}

export interface Tab {
  id: string;
  conversationId: string | null;
  title: string;
  messages: Message[];
  input: string;
  isStreaming: boolean;
  streamingContent: string;
  reasoningEvents: ReasoningEvent[];
  settings: ChatSettings | null;
}

interface TabsStore {
  // State
  tabs: Tab[];
  activeTabId: string | null;

  // Actions
  addTab: (tab: Omit<Tab, 'id'>) => string;
  removeTab: (tabId: string) => void;
  switchTab: (tabId: string) => void;
  updateTab: (tabId: string, updates: Partial<Tab>) => void;
  findTabByConversationId: (conversationId: string) => Tab | undefined;
  getActiveTab: () => Tab | undefined;
  closeAllTabs: () => void;
}

// ============================================================================
// Store
// ============================================================================

export const useTabsStore = create<TabsStore>((set, get) => ({
  tabs: [
    {
      id: 'tab-1',
      conversationId: null,
      title: 'New Chat',
      messages: [],
      input: '',
      isStreaming: false,
      streamingContent: '',
      reasoningEvents: [],
      settings: null,
    },
  ],
  activeTabId: 'tab-1',

  // Add a new tab
  addTab: (tab) => {
    const id = `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const newTab: Tab = { id, ...tab };

    set((state) => ({
      tabs: [...state.tabs, newTab],
      activeTabId: id,
    }));

    return id;
  },

  // Remove a tab
  removeTab: (tabId) => {
    set((state) => {
      const tabs = state.tabs.filter((t) => t.id !== tabId);

      if (tabs.length === 0) {
        const newTabId = `tab-${Date.now()}`;
        return {
          tabs: [
            {
              id: newTabId,
              conversationId: null,
              title: 'New Chat',
              messages: [],
              input: '',
              isStreaming: false,
              streamingContent: '',
              reasoningEvents: [],
              settings: null,
            },
          ],
          activeTabId: newTabId,
        };
      }

      // If we closed the active tab, switch to the rightmost tab
      let newActiveTabId = state.activeTabId;
      if (tabId === state.activeTabId) {
        const tabIndex = state.tabs.findIndex((t) => t.id === tabId);
        // Try to switch to tab on the right, or fall back to the left
        const nextTab = tabs[Math.min(tabIndex, tabs.length - 1)];
        newActiveTabId = nextTab.id;
      }

      return {
        tabs,
        activeTabId: newActiveTabId,
      };
    });
  },

  // Switch to a tab
  switchTab: (tabId) => {
    set({ activeTabId: tabId });
  },

  // Update tab data
  updateTab: (tabId, updates) => {
    set((state) => ({
      tabs: state.tabs.map((tab) => (tab.id === tabId ? { ...tab, ...updates } : tab)),
    }));
  },

  // Find tab by conversation ID
  findTabByConversationId: (conversationId) => {
    const state = get();
    return state.tabs.find((tab) => tab.conversationId === conversationId);
  },

  // Get the active tab
  getActiveTab: () => {
    const state = get();
    return state.tabs.find((tab) => tab.id === state.activeTabId);
  },

  closeAllTabs: () => {
    const newTabId = `tab-${Date.now()}`;
    set({
      tabs: [
        {
          id: newTabId,
          conversationId: null,
          title: 'New Chat',
          messages: [],
          input: '',
          isStreaming: false,
          streamingContent: '',
          reasoningEvents: [],
          settings: null,
        },
      ],
      activeTabId: newTabId,
    });
  },
}));

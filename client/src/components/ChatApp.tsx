/**
 * ChatApp - Main application component.
 *
 * Orchestrates the entire chat interface with multi-tab support, conversation management,
 * streaming responses, and keyboard shortcuts.
 *
 * Features:
 * - Multi-tab support for managing multiple conversations simultaneously
 * - Browser-style tabs with close buttons
 * - Each tab maintains its own conversation state
 * - Conversation list with search and filtering
 * - Settings panel for model and routing configuration
 * - Keyboard shortcuts for navigation and focus management
 */

import { useMemo, useEffect, useRef, useCallback, useState } from 'react';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { useAPIClient } from '../contexts/APIClientContext';
import { useModels } from '../hooks/useModels';
import { useConversations } from '../hooks/useConversations';
import { useLoadConversation } from '../hooks/useLoadConversation';
import { useKeyboardShortcuts, createCrossPlatformShortcut } from '../hooks/useKeyboardShortcuts';
import { ChatLayout, type ChatLayoutRef } from './ChatLayout';
import { AppLayout } from './layout/AppLayout';
import { SettingsPanel } from './settings/SettingsPanel';
import { ConversationList, type ConversationListRef } from './conversations/ConversationList';
import { TabBar } from './tabs/TabBar';
import { useChatStore } from '../store/chat-store';
import { useTabsStore } from '../store/tabs-store';
import { useConversationsStore, useViewFilter, useSearchQuery } from '../store/conversations-store';
import { processSearchResults } from '../lib/search-utils';
import type { MessageSearchResult } from '../lib/api-client';

export function ChatApp(): JSX.Element {
  const { client } = useAPIClient();
  const { content, isStreaming, error, reasoningEvents, usage, sendMessage, cancel, clear } =
    useStreamingChat(client);
  const wasStreamingRef = useRef(false);

  // Refs for keyboard shortcuts
  const conversationListRef = useRef<ConversationListRef>(null);
  const chatLayoutRef = useRef<ChatLayoutRef>(null);

  // Search results state
  const [searchResults, setSearchResults] = useState<MessageSearchResult[] | null>(null);

  // Fetch available models
  const { models, isLoading: isLoadingModels } = useModels(client);

  // Get settings from chat store
  const settings = useChatStore((state) => state.settings);

  // Tabs state
  const tabs = useTabsStore((state) => state.tabs);
  const activeTabId = useTabsStore((state) => state.activeTabId);
  const updateTab = useTabsStore((state) => state.updateTab);
  const addTab = useTabsStore((state) => state.addTab);
  const findTabByConversationId = useTabsStore((state) => state.findTabByConversationId);
  const switchTab = useTabsStore((state) => state.switchTab);

  // Get active tab
  const activeTab = tabs.find((tab) => tab.id === activeTabId);

  // Conversation management
  const {
    conversations,
    isLoading: conversationsLoading,
    error: conversationsError,
    selectedConversationId,
    fetchConversations,
    deleteConversation,
    archiveConversation,
    updateConversationTitle,
    selectConversation,
  } = useConversations(client);

  // View filter and search query from store
  const viewFilter = useViewFilter();
  const searchQuery = useSearchQuery();
  const setViewFilter = useConversationsStore((state) => state.setViewFilter);
  const setSearchQuery = useConversationsStore((state) => state.setSearchQuery);

  // Filter conversations based on active/archived view
  const filteredConversations = useMemo(() => {
    return conversations.filter((conv) => {
      const isArchived = conv.archived_at !== null;
      const isActive = !isArchived;

      if (viewFilter === 'active') {
        return isActive;
      } else {
        return isArchived;
      }
    });
  }, [conversations, viewFilter]);

  // Process search results into conversation list
  const searchResultConversations = useMemo(
    () => processSearchResults(searchResults, conversations),
    [searchResults, conversations],
  );

  // Display conversations: search results when searching, filtered list otherwise
  const displayConversations = searchResults ? searchResultConversations || [] : filteredConversations;

  // Clear search results when search query is cleared
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setSearchResults(null);
    }
  }, [searchQuery]);

  // Load conversation history
  const {
    isLoading: isLoadingHistory,
    loadConversation,
  } = useLoadConversation(client);

  // Build messages array for display from active tab
  const messages = useMemo(() => {
    if (!activeTab) return [];

    const msgs = [...activeTab.messages];

    // Add current streaming message if there's content or it's streaming
    if ((content || isStreaming) && activeTab.isStreaming) {
      msgs.push({
        role: 'assistant',
        content: content || '',
        reasoningEvents: reasoningEvents,
        usage: usage, // Include usage data from streaming
      });
    }

    return msgs;
  }, [activeTab, content, isStreaming, reasoningEvents, usage]);

  const handleSendMessage = async (userMessage: string) => {
    if (!activeTab) return;

    // Add user message to active tab history
    const updatedMessages = [
      ...activeTab.messages,
      {
        role: 'user' as const,
        content: userMessage,
      },
    ];

    updateTab(activeTab.id, {
      messages: updatedMessages,
      input: '',
      isStreaming: true,
    });

    // Determine temperature: gpt-5 models require temp=1
    const isGPT5Model = settings.model.toLowerCase().startsWith('gpt-5');
    const temperature = isGPT5Model ? 1.0 : settings.temperature;

    // Send to API with current settings
    const conversationId = await sendMessage(userMessage, {
      model: settings.model,
      routingMode: settings.routingMode,
      temperature: temperature,
      systemMessage: settings.systemPrompt || undefined,
      conversationId: activeTab.conversationId,
    });

    // Update tab with conversation ID if this was a new conversation
    if (conversationId && !activeTab.conversationId) {
      updateTab(activeTab.id, { conversationId });
      // Refresh conversation list to show the new conversation
      fetchConversations();
      // Select the new conversation in the sidebar
      selectConversation(conversationId);
    }
  };

  const handleCancel = () => {
    if (!activeTab) return;

    cancel();

    if (content) {
      const updatedMessages = [
        ...activeTab.messages,
        {
          role: 'assistant' as const,
          content: content + ' [cancelled]',
          reasoningEvents: reasoningEvents,
          usage: usage,
        },
      ];

      updateTab(activeTab.id, {
        messages: updatedMessages,
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });
    }

    // Auto-focus chat input after canceling
    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  };

  // When streaming completes, add the complete message to tab history
  useEffect(() => {
    if (!activeTab) return;

    if (wasStreamingRef.current && !isStreaming && content && !error) {
      const updatedMessages = [
        ...activeTab.messages,
        {
          role: 'assistant' as const,
          content: content,
          reasoningEvents: reasoningEvents,
          usage: usage,
        },
      ];

      updateTab(activeTab.id, {
        messages: updatedMessages,
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      // Clear the streaming content
      clear();

      // Auto-focus chat input after streaming completes
      // Use setTimeout to ensure DOM updates have completed
      setTimeout(() => {
        chatLayoutRef.current?.focusInput();
      }, 0);
    }

    wasStreamingRef.current = isStreaming;
  }, [isStreaming, content, error, reasoningEvents, clear, activeTab, updateTab]);

  // Handle conversation selection from sidebar
  const handleSelectConversation = useCallback(
    async (id: string) => {
      selectConversation(id);

      // Check if conversation is already open in a tab
      const existingTab = findTabByConversationId(id);

      if (existingTab) {
        // Switch to existing tab
        switchTab(existingTab.id);
      } else {
        // Load conversation and create new tab
        try {
          const history = await loadConversation(id);

          // Get conversation details for title
          const conversation = conversations.find((c) => c.id === id);
          const title = conversation?.title || 'Untitled';

          // Create new tab
          addTab({
            conversationId: id,
            title: title,
            messages: history,
            input: '',
            isStreaming: false,
            streamingContent: '',
            reasoningEvents: [],
          });
        } catch (err) {
          console.error('Failed to load conversation:', err);
        }
      }
    },
    [
      selectConversation,
      findTabByConversationId,
      switchTab,
      loadConversation,
      conversations,
      addTab,
    ],
  );

  // Handle new conversation (creates new tab)
  const handleNewConversation = useCallback(() => {
    addTab({
      conversationId: null,
      title: 'New Chat',
      messages: [],
      input: '',
      isStreaming: false,
      streamingContent: '',
      reasoningEvents: [],
    });
    selectConversation(null);

    // Auto-focus chat input for new conversation
    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  }, [addTab, selectConversation]);

  // Handle new tab button
  const handleNewTab = useCallback(() => {
    handleNewConversation();
  }, [handleNewConversation]);

  // Handle delete conversation
  const handleDeleteConversation = useCallback(
    async (id: string) => {
      await deleteConversation(id);

      // If conversation is open in a tab, close that tab
      const tabToClose = findTabByConversationId(id);
      if (tabToClose) {
        useTabsStore.getState().removeTab(tabToClose.id);
      }
    },
    [deleteConversation, findTabByConversationId],
  );

  // Handle archive conversation
  const handleArchiveConversation = useCallback(
    async (id: string) => {
      await archiveConversation(id);

      // If conversation is open in a tab, close that tab
      const tabToClose = findTabByConversationId(id);
      if (tabToClose) {
        useTabsStore.getState().removeTab(tabToClose.id);
      }
    },
    [archiveConversation, findTabByConversationId],
  );

  // Handle search
  const handleSearch = useCallback(
    async (query: string) => {
      try {
        const results = await client.searchMessages(query, {
          archiveFilter: viewFilter === 'archived' ? 'archived' : 'active',
          limit: 20,
        });
        setSearchResults(results.results);
      } catch (err) {
        console.error('Search failed:', err);
        setSearchResults(null);
      }
    },
    [client, viewFilter],
  );

  // Update tab input when it changes
  const handleInputChange = useCallback(
    (value: string) => {
      if (activeTab) {
        updateTab(activeTab.id, { input: value });
      }
    },
    [activeTab, updateTab],
  );

  // Handle close current tab
  const handleCloseCurrentTab = useCallback(() => {
    if (!activeTabId || tabs.length === 1) {
      // Don't close if it's the only tab
      return;
    }
    useTabsStore.getState().removeTab(activeTabId);
  }, [activeTabId, tabs.length]);

  // Keyboard shortcuts
  useKeyboardShortcuts([
    // Cmd+N (Ctrl+N on Windows/Linux): New conversation
    createCrossPlatformShortcut('n', handleNewConversation),

    // Cmd+W (Ctrl+W on Windows/Linux): Close current tab (unless only 1 tab)
    createCrossPlatformShortcut('w', handleCloseCurrentTab),

    // Cmd+Shift+F (Ctrl+Shift+F on Windows/Linux): Focus search box
    createCrossPlatformShortcut('f', () => conversationListRef.current?.focusSearch(), {
      shift: true,
    }),

    // Escape: Focus chat input (works even when focused in search/input fields)
    {
      key: 'Escape',
      handler: () => chatLayoutRef.current?.focusInput(),
      preventDefault: true,
      allowInInputFields: true,
    },
  ]);

  // Sync conversation titles to tabs when they change
  useEffect(() => {
    tabs.forEach((tab) => {
      if (tab.conversationId) {
        const conversation = conversations.find((c) => c.id === tab.conversationId);
        if (conversation) {
          const newTitle = conversation.title || 'Untitled';
          if (tab.title !== newTitle) {
            updateTab(tab.id, { title: newTitle });
          }
        }
      }
    });
  }, [conversations, tabs, updateTab]);

  return (
    <AppLayout
      conversationsSidebar={
        <ConversationList
          ref={conversationListRef}
          conversations={displayConversations}
          selectedConversationId={selectedConversationId}
          isLoading={conversationsLoading}
          error={conversationsError}
          viewFilter={viewFilter}
          searchQuery={searchQuery}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          onArchiveConversation={handleArchiveConversation}
          onUpdateTitle={updateConversationTitle}
          onRefresh={fetchConversations}
          onViewFilterChange={setViewFilter}
          onSearchQueryChange={setSearchQuery}
          onSearch={handleSearch}
        />
      }
      settingsSidebar={
        <SettingsPanel availableModels={models} isLoadingModels={isLoadingModels} />
      }
      tabBar={<TabBar onNewTab={handleNewTab} />}
    >
      <ChatLayout
        ref={chatLayoutRef}
        messages={messages}
        isStreaming={!!isStreaming && !!activeTab?.isStreaming}
        isLoadingHistory={isLoadingHistory}
        input={activeTab?.input || ''}
        onInputChange={handleInputChange}
        onSendMessage={handleSendMessage}
        onCancel={handleCancel}
      />
    </AppLayout>
  );
}

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
import { useLoadConversation, type DisplayMessage } from '../hooks/useLoadConversation';
import { useKeyboardShortcuts, createCrossPlatformShortcut } from '../hooks/useKeyboardShortcuts';
import { ChatLayout, type ChatLayoutRef } from './ChatLayout';
import { AppLayout } from './layout/AppLayout';
import { SettingsPanel } from './settings/SettingsPanel';
import { ConversationList, type ConversationListRef } from './conversations/ConversationList';
import { TabBar } from './tabs/TabBar';
import { useChatStore } from '../store/chat-store';
import { useTabsStore } from '../store/tabs-store';
import { useConversationsStore, useViewFilter, useSearchQuery } from '../store/conversations-store';
import { useConversationSettingsStore } from '../store/conversation-settings-store';
import { useToast } from '../store/toast-store';
import { processSearchResults } from '../lib/search-utils';
import type { MessageSearchResult } from '../lib/api-client';
import type { ReasoningEvent, Usage } from '../types/openai';

export function ChatApp(): JSX.Element {
  const { client } = useAPIClient();
  const { content, isStreaming, error, reasoningEvents, usage, sendMessage, regenerate, cancel, clear } =
    useStreamingChat(client);
  const wasStreamingRef = useRef(false);
  const toast = useToast();

  // Refs for keyboard shortcuts
  const conversationListRef = useRef<ConversationListRef>(null);
  const chatLayoutRef = useRef<ChatLayoutRef>(null);

  // Search results state
  const [searchResults, setSearchResults] = useState<MessageSearchResult[] | null>(null);

  // Sidebar states (global across all tabs)
  const [isConversationsOpen, setIsConversationsOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Fetch available models
  const { models, isLoading: isLoadingModels } = useModels(client);

  // Get settings from chat store
  const settings = useChatStore((state) => state.settings);
  const updateSettings = useChatStore((state) => state.updateSettings);

  // Conversation settings store
  const { getSettings, saveSettings } = useConversationSettingsStore();

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

  /**
   * Helper to start streaming an assistant response.
   * Ensures consistent streaming lifecycle for both sendMessage and regenerate.
   */
  const startStreamingResponse = useCallback(
    (messages: DisplayMessage[], extraUpdates: Partial<typeof activeTab> = {}) => {
      if (!activeTab) return;

      updateTab(activeTab.id, {
        messages,
        isStreaming: true,
        ...extraUpdates,
      });
    },
    [activeTab, updateTab],
  );

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

    // Start streaming response (clears input since message was just sent)
    startStreamingResponse(updatedMessages, { input: '' });

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

    if (conversationId && !activeTab.conversationId) {
      updateTab(activeTab.id, { conversationId });
      fetchConversations();
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

      // Reload conversation from database to get sequence numbers for new messages
      // This enables delete/regenerate/branch buttons on newly saved messages
      if (activeTab.conversationId) {
        loadConversation(activeTab.conversationId)
          .then((history) => {
            updateTab(activeTab.id, { messages: history });
          })
          .catch((err) => {
            console.error('Failed to reload conversation after streaming:', err);
            // Don't show error to user - messages are already displayed, just without sequence numbers
          });
      }

      // Auto-focus chat input after streaming completes
      // Use setTimeout to ensure DOM updates have completed
      setTimeout(() => {
        chatLayoutRef.current?.focusInput();
      }, 0);
    }

    wasStreamingRef.current = isStreaming;
  }, [isStreaming, content, error, reasoningEvents, clear, activeTab, updateTab, loadConversation]);

  // Handle conversation selection from sidebar
  const handleSelectConversation = useCallback(
    async (id: string) => {
      selectConversation(id);

      const existingTab = findTabByConversationId(id);

      if (existingTab) {
        switchTab(existingTab.id);
      } else {
        try {
          const history = await loadConversation(id);

          const conversation = conversations.find((c) => c.id === id);
          const title = conversation?.title || 'Untitled';

          addTab({
            conversationId: id,
            title: title,
            messages: history,
            input: '',
            isStreaming: false,
            streamingContent: '',
            reasoningEvents: [],
            settings: null,
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

  const handleNewConversation = useCallback(() => {
    addTab({
      conversationId: null,
      title: 'New Chat',
      messages: [],
      input: '',
      isStreaming: false,
      streamingContent: '',
      reasoningEvents: [],
      settings: null,
    });
    selectConversation(null);

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

  // Handle delete message (and all subsequent messages)
  const handleDeleteMessage = useCallback(
    async (messageIndex: number) => {
      if (!activeTab?.conversationId) return;

      const message = activeTab.messages[messageIndex];
      if (!message?.sequenceNumber) {
        console.warn('Cannot delete message without sequence number');
        return;
      }

      try {
        // Delete from API
        await client.deleteMessage(activeTab.conversationId, message.sequenceNumber);

        // Update local state: remove this message and all after it
        const updatedMessages = activeTab.messages.slice(0, messageIndex);
        updateTab(activeTab.id, { messages: updatedMessages });

        // Refresh conversation list to update message counts
        await fetchConversations();
      } catch (err) {
        console.error('Failed to delete message:', err);
        toast.error('Failed to delete message');
      }
    },
    [activeTab, client, updateTab, fetchConversations, toast],
  );

  // Handle regenerate message (delete assistant message and generate new response)
  const handleRegenerateMessage = useCallback(
    async (messageIndex: number) => {
      if (!activeTab?.conversationId) return;

      const message = activeTab.messages[messageIndex];
      if (!message?.sequenceNumber) {
        console.warn('Cannot regenerate message without sequence number');
        return;
      }

      try {
        // Delete the assistant message (and all after it) from API
        await client.deleteMessage(activeTab.conversationId, message.sequenceNumber);

        // Update local state: remove this message and all after it, start streaming response
        const updatedMessages = activeTab.messages.slice(0, messageIndex);
        startStreamingResponse(updatedMessages);

        // Determine temperature: gpt-5 models require temp=1
        const isGPT5Model = settings.model.toLowerCase().startsWith('gpt-5');
        const temperature = isGPT5Model ? 1.0 : settings.temperature;

        // Use hook's regenerate method (sends empty messages array, API uses conversation history)
        await regenerate({
          model: settings.model,
          routingMode: settings.routingMode,
          temperature: temperature,
          conversationId: activeTab.conversationId,
        });

        // Refresh conversations list
        await fetchConversations();
      } catch (err) {
        console.error('Failed to regenerate message:', err);
        toast.error('Failed to regenerate message');
      }
    },
    [activeTab, client, startStreamingResponse, settings, regenerate, fetchConversations, toast],
  );

  // Handle branch conversation (create new conversation from this point)
  const handleBranchConversation = useCallback(
    async (messageIndex: number) => {
      if (!activeTab?.conversationId) return;

      const message = activeTab.messages[messageIndex];
      if (!message?.sequenceNumber) {
        console.warn('Cannot branch from message without sequence number');
        return;
      }

      try {
        // Create branched conversation via API
        const branchedConversation = await client.branchConversation(
          activeTab.conversationId,
          message.sequenceNumber,
        );

        // Refresh conversation list to show new conversation
        await fetchConversations();

        // Open branched conversation in a new tab
        const history = branchedConversation.messages.map((msg) => {
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
            content: msg.content || '',
            reasoningEvents: msg.reasoning_events
              ? (msg.reasoning_events as unknown as ReasoningEvent[])
              : undefined,
            usage,
          };
        });

        addTab({
          conversationId: branchedConversation.id,
          title: branchedConversation.title || 'Branched Conversation',
          messages: history,
          input: '',
          isStreaming: false,
          streamingContent: '',
          reasoningEvents: [],
          settings: null,
        });

        // Select the newly branched conversation in the sidebar
        selectConversation(branchedConversation.id);

        toast.success('Conversation branched successfully');
      } catch (err) {
        console.error('Failed to branch conversation:', err);
        toast.error('Failed to branch conversation');
      }
    },
    [activeTab, client, fetchConversations, addTab, selectConversation, toast],
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

  // Handle toggle settings (global state)
  const handleToggleSettings = useCallback(() => {
    setIsSettingsOpen(!isSettingsOpen);
  }, [isSettingsOpen]);

  // Handle toggle conversations sidebar
  const handleToggleConversations = useCallback(() => {
    setIsConversationsOpen(!isConversationsOpen);
  }, [isConversationsOpen]);

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

  // Restore settings when switching tabs (only when tab ID changes)
  useEffect(() => {
    if (!activeTab) return;

    if (activeTab.settings) {
      updateSettings(activeTab.settings);
    } else if (activeTab.conversationId) {
      const conversationSettings = getSettings(activeTab.conversationId);
      updateSettings(conversationSettings);
    } else {
      const defaultSettings = getSettings(null);
      updateSettings(defaultSettings);
    }
  }, [activeTab?.id]);

  // Save settings whenever they change
  useEffect(() => {
    if (!activeTab) return;

    if (activeTab.conversationId) {
      saveSettings(activeTab.conversationId, settings);
    } else {
      updateTab(activeTab.id, { settings });
    }
  }, [settings]);

  // When new conversation gets an ID, migrate settings
  useEffect(() => {
    if (activeTab?.conversationId && activeTab.settings) {
      saveSettings(activeTab.conversationId, activeTab.settings);
      updateTab(activeTab.id, { settings: null });
    }
  }, [activeTab?.conversationId, activeTab?.settings]);

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
      isConversationsOpen={isConversationsOpen}
      tabBar={
        <TabBar
          onNewTab={handleNewTab}
          isSettingsOpen={isSettingsOpen}
          onToggleSettings={handleToggleSettings}
          isConversationsOpen={isConversationsOpen}
          onToggleConversations={handleToggleConversations}
        />
      }
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
        isSettingsOpen={isSettingsOpen}
        settingsPanel={
          <SettingsPanel availableModels={models} isLoadingModels={isLoadingModels} />
        }
        onDeleteMessage={handleDeleteMessage}
        onRegenerateMessage={handleRegenerateMessage}
        onBranchConversation={handleBranchConversation}
      />
    </AppLayout>
  );
}

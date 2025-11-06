/**
 * StreamingDemo component with tabs support.
 *
 * Milestone 10: Multi-tab support for managing multiple conversations simultaneously.
 * - Browser-style tabs with close buttons
 * - Each tab maintains its own conversation state
 * - Switch between tabs seamlessly
 * - New conversation creates new tab
 */

import { useMemo, useEffect, useRef, useCallback } from 'react';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { useAPIClient } from '../contexts/APIClientContext';
import { useModels } from '../hooks/useModels';
import { useConversations } from '../hooks/useConversations';
import { useLoadConversation } from '../hooks/useLoadConversation';
import { ChatLayout } from './ChatLayout';
import { AppLayout } from './layout/AppLayout';
import { SettingsPanel } from './settings/SettingsPanel';
import { ConversationList } from './conversations/ConversationList';
import { TabBar } from './tabs/TabBar';
import { useChatStore } from '../store/chat-store';
import { useTabsStore } from '../store/tabs-store';

export function StreamingDemo(): JSX.Element {
  const { client } = useAPIClient();
  const { content, isStreaming, error, reasoningEvents, sendMessage, cancel, clear } =
    useStreamingChat(client);
  const wasStreamingRef = useRef(false);

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
    updateConversationTitle,
    selectConversation,
  } = useConversations(client);

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
      });
    }

    return msgs;
  }, [activeTab, content, isStreaming, reasoningEvents]);

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
    }
  };

  const handleCancel = () => {
    if (!activeTab) return;

    cancel();

    // If there was partial content, add it to tab history
    if (content) {
      const updatedMessages = [
        ...activeTab.messages,
        {
          role: 'assistant' as const,
          content: content + ' [cancelled]',
          reasoningEvents: reasoningEvents,
        },
      ];

      updateTab(activeTab.id, {
        messages: updatedMessages,
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });
    }
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

      // Update conversation ID if this was a new conversation
      if (!activeTab.conversationId) {
        // The conversationId would come from the API response
        // We'll need to extract it from the streaming response
        // For now, this will be handled separately
      }
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

  // Update tab input when it changes
  const handleInputChange = useCallback(
    (value: string) => {
      if (activeTab) {
        updateTab(activeTab.id, { input: value });
      }
    },
    [activeTab, updateTab],
  );

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
          conversations={conversations}
          selectedConversationId={selectedConversationId}
          isLoading={conversationsLoading}
          error={conversationsError}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          onUpdateTitle={updateConversationTitle}
          onRefresh={fetchConversations}
        />
      }
      settingsSidebar={
        <SettingsPanel availableModels={models} isLoadingModels={isLoadingModels} />
      }
      tabBar={<TabBar onNewTab={handleNewTab} />}
    >
      <ChatLayout
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

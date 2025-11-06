/**
 * StreamingDemo component - main application.
 *
 * Milestone 9: Conversation management with sidebar.
 * - Conversations list sidebar (collapsible/resizable)
 * - Settings panel (collapsible)
 * - Load conversation history
 * - Delete/edit conversations
 */

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { useAPIClient } from '../contexts/APIClientContext';
import { useModels } from '../hooks/useModels';
import { useConversations } from '../hooks/useConversations';
import { useLoadConversation } from '../hooks/useLoadConversation';
import { ChatLayout, type Message } from './ChatLayout';
import { AppLayout } from './layout/AppLayout';
import { SettingsPanel } from './settings/SettingsPanel';
import { ConversationList } from './conversations/ConversationList';
import { useChatStore } from '../store/chat-store';

export function StreamingDemo(): JSX.Element {
  const { client } = useAPIClient();
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const { content, isStreaming, error, reasoningEvents, sendMessage, cancel, clear } =
    useStreamingChat(client);
  const wasStreamingRef = useRef(false);

  // Fetch available models
  const { models, isLoading: isLoadingModels } = useModels(client);

  // Get settings and conversation ID from chat store
  const settings = useChatStore((state) => state.settings);
  const conversationId = useChatStore((state) => state.conversationId);
  const setConversationId = useChatStore((state) => state.setConversationId);
  const newConversation = useChatStore((state) => state.newConversation);

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
    clearMessages,
  } = useLoadConversation(client);

  // Build messages array for display
  const messages = useMemo(() => {
    const msgs = [...conversationHistory];

    // Add current streaming message if there's content or it's streaming
    if (content || isStreaming) {
      msgs.push({
        role: 'assistant',
        content: content || '',
        reasoningEvents: reasoningEvents,
      });
    }

    return msgs;
  }, [conversationHistory, content, isStreaming, reasoningEvents]);

  const handleSendMessage = async (userMessage: string) => {
    // Add user message to history
    setConversationHistory((prev) => [
      ...prev,
      {
        role: 'user',
        content: userMessage,
      },
    ]);

    // Clear input
    setInput('');

    // Determine temperature: gpt-5 models require temp=1
    const isGPT5Model = settings.model.toLowerCase().startsWith('gpt-5');
    const temperature = isGPT5Model ? 1.0 : settings.temperature;

    // Send to API with current settings
    await sendMessage(userMessage, {
      model: settings.model,
      routingMode: settings.routingMode,
      temperature: temperature,
      // Note: maxTokens intentionally omitted - let API use model's max context
      systemMessage: settings.systemPrompt || undefined,
    });
  };

  const handleCancel = () => {
    cancel();

    // If there was partial content, add it to history
    if (content) {
      setConversationHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: content + ' [cancelled]',
          reasoningEvents: reasoningEvents,
        },
      ]);
    }
  };

  // When streaming completes, add the complete message to history
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming && content && !error) {
      setConversationHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: content,
          reasoningEvents: reasoningEvents,
        },
      ]);
      // Clear the streaming content after adding to history to prevent duplication
      clear();
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming, content, error, reasoningEvents, clear]);

  // When a new conversation is created (conversationId changes from null to a value),
  // refresh the conversation list and select the new conversation
  const previousConversationIdRef = useRef<string | null>(null);
  const isFirstRenderRef = useRef(true);

  useEffect(() => {
    const previousId = previousConversationIdRef.current;
    const currentId = conversationId;

    // Skip first render to avoid false positive on mount
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false;
      previousConversationIdRef.current = currentId;
      return;
    }

    // Detect new conversation creation: previousId was null, currentId is not null
    if (previousId === null && currentId !== null) {
      console.log('[StreamingDemo] New conversation created, refreshing list:', currentId);
      // Refresh conversation list to show the new conversation
      fetchConversations();
      // Select the new conversation in the sidebar
      selectConversation(currentId);
    }

    previousConversationIdRef.current = currentId;
  }, [conversationId, fetchConversations, selectConversation]);

  // Handle conversation selection
  const handleSelectConversation = useCallback(
    async (id: string) => {
      selectConversation(id);

      try {
        // Load conversation history
        const history = await loadConversation(id);
        setConversationHistory(history);

        // Set conversation ID in chat store for continuing the conversation
        setConversationId(id);
      } catch (err) {
        console.error('Failed to load conversation:', err);
        // Error is already set in useLoadConversation hook
      }
    },
    [selectConversation, loadConversation, setConversationId],
  );

  // Handle new conversation
  const handleNewConversation = useCallback(() => {
    newConversation();
    clearMessages();
    setConversationHistory([]);
    selectConversation(null);
  }, [newConversation, clearMessages, selectConversation]);

  // Handle delete conversation
  const handleDeleteConversation = useCallback(
    async (id: string) => {
      await deleteConversation(id);

      // If we deleted the current conversation, start a new one
      if (id === conversationId) {
        handleNewConversation();
      }
    },
    [deleteConversation, conversationId, handleNewConversation],
  );

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
    >
      <ChatLayout
        messages={messages}
        isStreaming={isStreaming}
        isLoadingHistory={isLoadingHistory}
        input={input}
        onInputChange={setInput}
        onSendMessage={handleSendMessage}
        onCancel={handleCancel}
      />
    </AppLayout>
  );
}

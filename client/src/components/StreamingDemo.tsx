/**
 * StreamingDemo component - main application.
 *
 * Milestone 5: Split-pane layout with settings panel and routing controls.
 * - Settings panel with model selector, temperature, etc.
 * - Routing mode selector (Passthrough/Reasoning/Auto)
 * - Clean chat interface
 */

import { useState, useMemo, useEffect, useRef } from 'react';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { useAPIClient } from '../contexts/APIClientContext';
import { useModels } from '../hooks/useModels';
import { ChatLayout, type Message } from './ChatLayout';
import { SplitLayout } from './layout/SplitLayout';
import { SettingsPanel } from './settings/SettingsPanel';
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

  // Get settings from store
  const settings = useChatStore((state) => state.settings);

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

  return (
    <SplitLayout
      sidebar={
        <SettingsPanel availableModels={models} isLoadingModels={isLoadingModels} />
      }
    >
      <ChatLayout
        messages={messages}
        isStreaming={isStreaming}
        input={input}
        onInputChange={setInput}
        onSendMessage={handleSendMessage}
        onCancel={handleCancel}
      />
    </SplitLayout>
  );
}

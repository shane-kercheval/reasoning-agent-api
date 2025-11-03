/**
 * Demo component showing streaming chat in action.
 *
 * Milestone 3: Minimal, clean chat interface with new UI components.
 * - ChatGPT-inspired minimal design
 * - Clean typography and spacing
 * - Smooth interactions
 */

import { useState, useMemo, useEffect, useRef } from 'react';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { useAPIClient } from '../contexts/APIClientContext';
import { RoutingMode, APIDefaults } from '../constants';
import { ChatLayout, type Message } from './ChatLayout';

export function StreamingDemo(): JSX.Element {
  const { client } = useAPIClient();
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const { content, isStreaming, error, reasoningEvents, sendMessage, cancel, clear } =
    useStreamingChat(client);
  const wasStreamingRef = useRef(false);

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

    // Send to API
    await sendMessage(userMessage, {
      model: APIDefaults.MODEL,
      routingMode: RoutingMode.REASONING, // Use reasoning mode to see reasoning events
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
    <ChatLayout
      messages={messages}
      isStreaming={isStreaming}
      onSendMessage={handleSendMessage}
      onCancel={handleCancel}
    />
  );
}

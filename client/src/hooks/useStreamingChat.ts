/**
 * React hook for streaming chat completions.
 *
 * Manages streaming state, response accumulation, and cancellation.
 * Uses Zustand store for global state management.
 * Includes proper cleanup to prevent memory leaks.
 */

import { useCallback, useRef, useEffect } from 'react';
import type { Message, MessageRole, ReasoningEvent } from '../types/openai';
import { isSSEDone, isChatCompletionChunk, isConversationMetadata } from '../types/openai';
import type { APIClient, ChatCompletionOptions } from '../lib/api-client';
import { useChatStore } from '../store/chat-store';

/**
 * State for streaming chat (now from Zustand store).
 */
export interface StreamingChatState {
  /** Current accumulated content from assistant */
  content: string;
  /** Reasoning events from the stream */
  reasoningEvents: ReasoningEvent[];
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Error message if request failed */
  error: string | null;
  /** Conversation ID for stateful mode */
  conversationId: string | null;
}

/**
 * Actions for streaming chat.
 */
export interface StreamingChatActions {
  /** Send a message and start streaming response */
  sendMessage: (content: string, options?: SendMessageOptions) => Promise<void>;
  /** Cancel current streaming request */
  cancel: () => void;
  /** Clear current state */
  clear: () => void;
  /** Start a new conversation */
  newConversation: () => void;
}

/**
 * Options for sending a message.
 */
export interface SendMessageOptions {
  /** System message */
  systemMessage?: string;
  /** Model to use */
  model?: string;
  /** Temperature (0-2) */
  temperature?: number;
  /** Max tokens in response */
  maxTokens?: number;
  /** Routing mode */
  routingMode?: ChatCompletionOptions['routingMode'];
}

/**
 * Hook for streaming chat completions.
 *
 * @param apiClient - Configured API client
 * @returns Streaming chat state and actions
 *
 * @example
 * ```typescript
 * const { client } = useAPIClient();
 * const { content, isStreaming, reasoningEvents, sendMessage, cancel } = useStreamingChat(client);
 *
 * return (
 *   <div>
 *     {isStreaming && <div>Streaming...</div>}
 *     <div>{content}</div>
 *     {reasoningEvents.map((event, i) => (
 *       <div key={i}>{event.type}: {JSON.stringify(event.metadata)}</div>
 *     ))}
 *     <button onClick={() => sendMessage('Hello!')}>Send</button>
 *     {isStreaming && <button onClick={cancel}>Cancel</button>}
 *   </div>
 * );
 * ```
 */
export function useStreamingChat(apiClient: APIClient): StreamingChatState & StreamingChatActions {
  // Get state and actions from Zustand store
  const conversationId = useChatStore((state) => state.conversationId);
  const streaming = useChatStore((state) => state.streaming);
  const {
    setConversationId,
    startStreaming,
    appendContent,
    addReasoningEvent,
    setError,
    stopStreaming,
    clearStreaming,
    newConversation,
  } = useChatStore();

  // AbortController for cancellation (not in store - needs to be per-hook instance)
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount - abort any in-flight requests
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  /**
   * Send a message and stream the response.
   */
  const sendMessage = useCallback(
    async (userMessage: string, options?: SendMessageOptions) => {
      // Prevent multiple concurrent requests
      if (streaming.isStreaming) {
        return;
      }

      // Create abort controller
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Reset state and start streaming
      clearStreaming();
      startStreaming();

      console.log('[useStreamingChat] Sending message with conversationId:', conversationId);

      try {
        // Build messages array
        const messages: Message[] = [];

        // Add system message if provided
        if (options?.systemMessage) {
          messages.push({
            role: 'system' as MessageRole,
            content: options.systemMessage,
          });
        }

        // Add user message
        messages.push({
          role: 'user' as MessageRole,
          content: userMessage,
        });

        // Stream completion
        for await (const chunk of apiClient.streamChatCompletion(
          {
            model: options?.model || 'gpt-4o-mini',
            messages,
            temperature: options?.temperature,
            max_tokens: options?.maxTokens,
            stream: true,
          },
          {
            signal: abortController.signal,
            conversationId: conversationId,
            routingMode: options?.routingMode,
          },
        )) {
          // Check for [DONE] marker
          if (isSSEDone(chunk)) {
            break;
          }

          // Check for conversation metadata
          if (isConversationMetadata(chunk)) {
            console.log('[useStreamingChat] Received conversation metadata:', chunk.conversationId);
            // Store conversation ID in Zustand for next request
            setConversationId(chunk.conversationId);
            console.log('[useStreamingChat] Stored conversation ID in Zustand');
            continue;
          }

          // Process chunk
          if (isChatCompletionChunk(chunk)) {
            const choice = chunk.choices[0];
            if (!choice) continue;

            const delta = choice.delta;

            // Accumulate content
            if (delta.content) {
              appendContent(delta.content);
            }

            // Add reasoning events
            if (delta.reasoning_event) {
              addReasoningEvent(delta.reasoning_event);
            }
          }
        }
      } catch (err) {
        // Handle abort separately
        if (err instanceof Error && err.name === 'AbortError') {
          setError('Request cancelled');
        } else {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        stopStreaming();
        abortControllerRef.current = null;
      }
    },
    [
      apiClient,
      streaming.isStreaming,
      conversationId,
      setConversationId,
      clearStreaming,
      startStreaming,
      appendContent,
      addReasoningEvent,
      setError,
      stopStreaming,
    ],
  );

  /**
   * Cancel the current streaming request.
   */
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  /**
   * Clear is now an alias for clearStreaming from store.
   */
  const clear = clearStreaming;

  return {
    // State (from store)
    content: streaming.content,
    reasoningEvents: streaming.reasoningEvents,
    isStreaming: streaming.isStreaming,
    error: streaming.error,
    conversationId,

    // Actions
    sendMessage,
    cancel,
    clear,
    newConversation,
  };
}

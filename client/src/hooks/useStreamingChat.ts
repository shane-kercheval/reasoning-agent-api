/**
 * React hook for streaming chat completions.
 *
 * Manages streaming state, response accumulation, and cancellation.
 * Includes proper cleanup to prevent memory leaks.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, MessageRole, ReasoningEvent } from '../types/openai';
import { isSSEDone, isChatCompletionChunk } from '../types/openai';
import type { APIClient, ChatCompletionOptions } from '../lib/api-client';

/**
 * State for streaming chat.
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
  const [content, setContent] = useState('');
  const [reasoningEvents, setReasoningEvents] = useState<ReasoningEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);

  // AbortController for cancellation
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
      if (isStreaming) {
        return;
      }

      // Create abort controller
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Reset state
      setContent('');
      setReasoningEvents([]);
      setError(null);
      setIsStreaming(true);

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
            conversationId: conversationId || undefined,
            routingMode: options?.routingMode,
          },
        )) {
          // Check for [DONE] marker
          if (isSSEDone(chunk)) {
            break;
          }

          // Process chunk
          if (isChatCompletionChunk(chunk)) {
            const choice = chunk.choices[0];
            if (!choice) continue;

            const delta = choice.delta;

            // Accumulate content
            if (delta.content) {
              setContent((prev) => prev + delta.content);
            }

            // Add reasoning events
            if (delta.reasoning_event) {
              setReasoningEvents((prev) => [...prev, delta.reasoning_event!]);
            }

            // Note: Conversation ID should be read from response headers (X-Conversation-ID)
            // Not implemented yet - using stateless mode for now
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
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [apiClient, isStreaming, conversationId],
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
   * Clear current state.
   */
  const clear = useCallback(() => {
    setContent('');
    setReasoningEvents([]);
    setError(null);
  }, []);

  /**
   * Start a new conversation.
   */
  const newConversation = useCallback(() => {
    setConversationId(null);
    clear();
  }, [clear]);

  return {
    // State
    content,
    reasoningEvents,
    isStreaming,
    error,
    conversationId,

    // Actions
    sendMessage,
    cancel,
    clear,
    newConversation,
  };
}

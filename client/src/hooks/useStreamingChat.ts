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

import type { Usage } from '../types/openai';

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
  /** Usage statistics from last response */
  usage: Usage | null;
}

/**
 * Actions for streaming chat.
 */
export interface StreamingChatActions {
  /** Send a message and start streaming response. Returns new conversation ID if created. */
  sendMessage: (content: string, options?: SendMessageOptions) => Promise<string | null>;
  /** Regenerate last assistant response without sending new user message. */
  regenerate: (options?: SendMessageOptions) => Promise<string | null>;
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
  /** Conversation ID for stateful mode */
  conversationId?: string | null;
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
   * Core streaming implementation shared by sendMessage and regenerate.
   * Handles the actual streaming, state updates, and error handling.
   *
   * @private
   */
  const streamCompletion = useCallback(
    async (
      messages: Message[],
      options?: SendMessageOptions,
    ): Promise<string | null> => {
      // Prevent multiple concurrent requests
      if (streaming.isStreaming) {
        return null;
      }

      // Create abort controller
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Reset state and start streaming
      clearStreaming();
      startStreaming();

      // Use conversation ID from options (for multi-tab support)
      const requestConversationId = options?.conversationId ?? null;
      let newConversationId: string | null = null;

      try {
        // Stream completion
        for await (const chunk of apiClient.streamChatCompletion(
          {
            model: options?.model || 'gpt-4o-mini',
            messages,
            temperature: options?.temperature,
            max_tokens: options?.maxTokens,
            stream: true,
            stream_options: { include_usage: true }, // Request usage data in final chunk
          },
          {
            signal: abortController.signal,
            conversationId: requestConversationId,
            routingMode: options?.routingMode,
          },
        )) {
          // Check for [DONE] marker
          if (isSSEDone(chunk)) {
            break;
          }

          // Check for conversation metadata
          if (isConversationMetadata(chunk)) {
            // Store the new conversation ID to return
            newConversationId = chunk.conversationId;
            // Also store in global store for backward compatibility
            setConversationId(chunk.conversationId);
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

            // Capture usage data (typically sent in final chunk)
            if (chunk.usage) {
              useChatStore.getState().setUsage(chunk.usage);
            }
          }
        }

        // Return the conversation ID (new one if created, or the one passed in)
        return newConversationId ?? requestConversationId;
      } catch (err) {
        // Handle abort separately
        if (err instanceof Error && err.name === 'AbortError') {
          setError('Request cancelled');
        } else {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
        return null;
      } finally {
        stopStreaming();
        abortControllerRef.current = null;
      }
    },
    [
      apiClient,
      streaming.isStreaming,
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
   * Send a message and stream the response.
   * Returns the conversation ID (either the one passed in or a new one from the API).
   */
  const sendMessage = useCallback(
    async (userMessage: string, options?: SendMessageOptions): Promise<string | null> => {
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

      // Stream using shared implementation
      return streamCompletion(messages, options);
    },
    [streamCompletion],
  );

  /**
   * Regenerate last assistant response without sending new user message.
   * Uses conversation history from the database.
   * Returns the conversation ID.
   */
  const regenerate = useCallback(
    async (options?: SendMessageOptions): Promise<string | null> => {
      // Empty messages array - API will use conversation history
      const messages: Message[] = [];

      // Stream using shared implementation
      return streamCompletion(messages, options);
    },
    [streamCompletion],
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
    usage: streaming.usage,

    // Actions
    sendMessage,
    regenerate,
    cancel,
    clear,
    newConversation,
  };
}

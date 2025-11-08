/**
 * Tab-aware streaming hook for chat completions.
 *
 * Provides tab-isolated streaming actions that update per-tab state in the tabs store.
 * Each tab can stream independently without interfering with other tabs.
 */

import { useEffect, useRef, useMemo, useCallback } from 'react';
import type { APIClient } from '../lib/api-client';
import { useTabsStore } from '../store/tabs-store';
import { useConversationSettingsStore } from '../store/conversation-settings-store';
import {
  isChatCompletionChunk,
  isConversationMetadata,
  isSSEDone,
  MessageRole,
  type ChatCompletionRequest,
  type ReasoningEvent,
  type Usage,
} from '../types/openai';

// ============================================================================
// Types
// ============================================================================

export interface SendMessageOptions {
  systemPrompt?: string;
  model?: string;
  temperature?: number;
  routingMode?: 'passthrough' | 'reasoning' | 'orchestration' | 'auto';
}

export interface TabStreamingActions {
  sendMessageForTab: (
    tabId: string,
    message: string,
    options?: SendMessageOptions
  ) => Promise<string | null>;
  regenerateForTab: (tabId: string, options?: SendMessageOptions) => Promise<string | null>;
  cancelStreamForTab: (tabId: string) => void;
}

// ============================================================================
// Hook
// ============================================================================

/**
 * Hook that provides tab-aware streaming actions.
 *
 * @param apiClient - API client for streaming requests
 * @returns Tab-aware streaming actions
 *
 * @example
 * ```tsx
 * const { sendMessageForTab, cancelStreamForTab, regenerateForTab } = useTabStreaming(client);
 *
 * // Send message in specific tab
 * await sendMessageForTab(activeTab.id, "Hello!");
 *
 * // Cancel stream in specific tab
 * cancelStreamForTab(activeTab.id);
 * ```
 */
export function useTabStreaming(apiClient: APIClient): TabStreamingActions {
  // Map of tabId → AbortController for stream cancellation
  const streamControllers = useRef<Map<string, AbortController>>(new Map());

  const { updateTab } = useTabsStore();
  const { getSettings, saveSettings } = useConversationSettingsStore();

  /**
   * Cleanup function for a specific tab's stream.
   */
  const cleanupTabStream = useCallback((tabId: string): void => {
    const controller = streamControllers.current.get(tabId);
    if (controller) {
      controller.abort();
      streamControllers.current.delete(tabId);
    }
  }, []);

  /**
   * Register cleanup function with tabs store and cleanup on unmount.
   */
  useEffect(() => {
    // Register cleanup function so removeTab can call it
    useTabsStore.setState({ streamCleanup: cleanupTabStream });

    // Capture ref value for cleanup
    const controllers = streamControllers.current;
    return () => {
      // Cleanup all active streams when hook unmounts
      controllers.forEach((controller) => controller.abort());
      controllers.clear();
    };
  }, [cleanupTabStream]);

  /**
   * Stream chat completion for a specific tab.
   */
  const streamCompletion = useCallback(async (
    tabId: string,
    messages: ChatCompletionRequest['messages'],
    options?: SendMessageOptions
  ): Promise<string | null> => {
    const tab = useTabsStore.getState().tabs.find((t) => t.id === tabId);
    if (!tab) {
      console.warn(`Tab ${tabId} not found, aborting stream`);
      return null;
    }

    // Get settings for this conversation (or defaults)
    const conversationSettings = getSettings(tab.conversationId);

    // Build request with options overriding conversation settings
    const model = options?.model ?? conversationSettings.model;
    const temperature = options?.temperature ?? conversationSettings.temperature;
    const routingMode = options?.routingMode ?? conversationSettings.routingMode;

    const request: ChatCompletionRequest = {
      model,
      temperature,
      messages,
      stream: true,
      stream_options: { include_usage: true },
    };

    // Create AbortController for this stream
    const abortController = new AbortController();
    streamControllers.current.set(tabId, abortController);

    // Initialize streaming state
    updateTab(tabId, {
      isStreaming: true,
      streamingContent: '',
      reasoningEvents: [],
      usage: null,
      streamError: null,
    });

    let accumulatedContent = '';
    const accumulatedReasoningEvents: ReasoningEvent[] = [];
    let streamUsage: Usage | null = null;
    let newConversationId: string | null = tab.conversationId;

    try {
      const stream = apiClient.streamChatCompletion(request, {
        signal: abortController.signal,
        conversationId: tab.conversationId,
        routingMode,
      });

      // Per-stream throttling (not shared across tabs)
      let lastUpdateTime = 0;
      const UPDATE_THROTTLE_MS = 50; // Update UI max every 50ms to reduce re-renders

      for await (const data of stream) {
        // Check if tab still exists
        const currentTab = useTabsStore.getState().tabs.find((t) => t.id === tabId);
        if (!currentTab) {
          console.warn(`Tab ${tabId} was deleted during streaming`);
          abortController.abort();
          break;
        }

        if (isConversationMetadata(data)) {
          newConversationId = data.conversationId;
          continue;
        }

        if (isSSEDone(data)) {
          break;
        }

        if (isChatCompletionChunk(data)) {
          const choice = data.choices[0];
          if (!choice) continue;

          let shouldUpdate = false;

          // Accumulate content
          if (choice.delta.content) {
            accumulatedContent += choice.delta.content;
            shouldUpdate = true;
          }

          // Accumulate reasoning events
          if (choice.delta.reasoning_event) {
            accumulatedReasoningEvents.push(choice.delta.reasoning_event);
            shouldUpdate = true;
          }

          // Capture usage data
          if (data.usage) {
            streamUsage = data.usage;
            shouldUpdate = true;
          }

          // Throttle updates to reduce re-renders (per-stream, not global)
          const now = Date.now();
          if (shouldUpdate && (now - lastUpdateTime) >= UPDATE_THROTTLE_MS) {
            updateTab(tabId, {
              streamingContent: accumulatedContent,
              reasoningEvents: accumulatedReasoningEvents,
              usage: streamUsage,
            });
            lastUpdateTime = now;
          }
        }
      }

      // Final update with any remaining content
      updateTab(tabId, {
        streamingContent: accumulatedContent,
        reasoningEvents: accumulatedReasoningEvents,
        usage: streamUsage,
      });

      // Stream completed successfully
      const finalTab = useTabsStore.getState().tabs.find((t) => t.id === tabId);
      if (!finalTab) {
        console.warn(`[useTabStreaming] Tab ${tabId} not found after streaming`);
        return newConversationId;
      }

      // Compute sequence number for assistant message (user message already has sequence number)
      // Find the highest sequence number in existing messages
      const maxSeq = finalTab.messages.reduce(
        (max, msg) => Math.max(max, msg.sequenceNumber ?? 0),
        0
      );
      const assistantSeq = maxSeq + 1;

      // Move streaming content to messages with computed sequence number
      const updatedMessages = [
        ...finalTab.messages,
        {
          role: 'assistant' as const,
          content: accumulatedContent,
          sequenceNumber: assistantSeq,
          reasoningEvents: accumulatedReasoningEvents.length > 0 ? accumulatedReasoningEvents : undefined,
          usage: streamUsage,
        },
      ];

      updateTab(tabId, {
        messages: updatedMessages,
        conversationId: newConversationId,
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
        usage: null,
      });

      // Save settings for this conversation
      if (newConversationId) {
        saveSettings(newConversationId, {
          model,
          temperature,
          routingMode,
          systemPrompt: options?.systemPrompt || conversationSettings.systemPrompt || '',
        });
      }

      return newConversationId;
    } catch (error) {
      // Handle abort separately
      if (error instanceof Error && error.name === 'AbortError') {
        updateTab(tabId, {
          isStreaming: false,
          streamingContent: '',
        });
        return newConversationId;
      }

      // Handle other errors
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Streaming error:', errorMessage);
      updateTab(tabId, {
        isStreaming: false,
        streamError: errorMessage,
      });
      return newConversationId;
    } finally {
      // Cleanup controller
      streamControllers.current.delete(tabId);
    }
  }, [apiClient, updateTab, getSettings, saveSettings]);

  /**
   * Send a message in a specific tab.
   */
  const sendMessageForTab = useCallback(async (
    tabId: string,
    message: string,
    options?: SendMessageOptions
  ): Promise<string | null> => {
    const tab = useTabsStore.getState().tabs.find((t) => t.id === tabId);
    if (!tab) {
      console.warn(`Tab ${tabId} not found`);
      return null;
    }

    // Get conversation settings for system prompt
    const conversationSettings = getSettings(tab.conversationId);
    const systemPrompt = options?.systemPrompt ?? conversationSettings.systemPrompt;

    // Build messages array - ONLY send new message
    // API will retrieve conversation history automatically via conversationId
    const messages: ChatCompletionRequest['messages'] = [];

    // Add system message if provided
    if (systemPrompt) {
      messages.push({
        role: MessageRole.System,
        content: systemPrompt,
      });
    }

    // Add new user message (API retrieves history automatically)
    messages.push({
      role: MessageRole.User,
      content: message,
    });

    // Compute sequence number for user message
    const maxSeq = tab.messages.reduce(
      (max, msg) => Math.max(max, msg.sequenceNumber ?? 0),
      0
    );
    const userSeq = maxSeq + 1;

    // Add user message to tab immediately with sequence number
    updateTab(tabId, {
      messages: [
        ...tab.messages,
        {
          role: 'user',
          content: message,
          sequenceNumber: userSeq,
        },
      ],
      input: '', // Clear input
    });

    // Stream completion
    return streamCompletion(tabId, messages, options);
  }, [updateTab, getSettings, streamCompletion]);

  /**
   * Regenerate the last assistant message in a specific tab.
   */
  const regenerateForTab = useCallback(async (
    tabId: string,
    options?: SendMessageOptions
  ): Promise<string | null> => {
    const tab = useTabsStore.getState().tabs.find((t) => t.id === tabId);
    if (!tab) {
      console.warn(`Tab ${tabId} not found`);
      return null;
    }

    if (tab.messages.length === 0) {
      console.warn('No messages to regenerate');
      return null;
    }

    // Get conversation settings for system prompt
    const conversationSettings = getSettings(tab.conversationId);
    const systemPrompt = options?.systemPrompt ?? conversationSettings.systemPrompt;

    // Build messages array - ONLY send system prompt
    // API will retrieve conversation history automatically via conversationId
    const messages: ChatCompletionRequest['messages'] = [];

    // Add system message if provided
    if (systemPrompt) {
      messages.push({
        role: MessageRole.System,
        content: systemPrompt,
      });
    }

    // Find last user message index to remove assistant messages after it
    let lastUserIndex = -1;
    for (let i = tab.messages.length - 1; i >= 0; i--) {
      if (tab.messages[i].role === 'user') {
        lastUserIndex = i;
        break;
      }
    }

    if (lastUserIndex === -1) {
      console.warn('No user message found to regenerate from');
      return null;
    }

    // Remove messages after last user message from UI
    updateTab(tabId, {
      messages: tab.messages.slice(0, lastUserIndex + 1),
    });

    // Stream completion (API retrieves history including last user message)
    return streamCompletion(tabId, messages, options);
  }, [updateTab, getSettings, streamCompletion]);

  /**
   * Cancel streaming for a specific tab.
   */
  const cancelStreamForTab = useCallback((tabId: string): void => {
    cleanupTabStream(tabId);
    updateTab(tabId, {
      isStreaming: false,
      streamingContent: '',
    });
  }, [cleanupTabStream, updateTab]);

  // Memoize the returned object to prevent recreating on every render
  return useMemo(
    () => ({
      sendMessageForTab,
      regenerateForTab,
      cancelStreamForTab,
    }),
    [sendMessageForTab, regenerateForTab, cancelStreamForTab]
  );
}

/**
 * MessageList component - renders list of chat messages.
 *
 * Memoized to prevent re-renders when unrelated props (like input) change.
 * Only re-renders when messages or streaming state actually changes.
 */

import * as React from 'react';
import { ChatMessage } from './ChatMessage';
import { StreamingIndicator } from './StreamingIndicator';
import type { ReasoningEvent, Usage } from '../../types/openai';

export interface Message {
  id?: string;  // UUID from database
  sequenceNumber?: number;  // Sequence number from database
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
  usage?: Usage | null;
}

export interface MessageListProps {
  messages: Message[];
  isStreaming: boolean;
  onDeleteMessage?: (messageIndex: number) => void;
  onRegenerateMessage?: (messageIndex: number) => void;
  onBranchConversation?: (messageIndex: number) => void;
}

/**
 * Memoized message list component.
 *
 * Only re-renders when messages, streaming state, or handlers change.
 * Does NOT re-render when unrelated props like input change.
 */
export const MessageList = React.memo<MessageListProps>(
  function MessageList({
    messages,
    isStreaming,
    onDeleteMessage,
    onRegenerateMessage,
    onBranchConversation,
  }) {
    return (
      <div className="mx-auto max-w-3xl py-8 overflow-x-hidden">
        <div className="space-y-6">
          {messages.map((message, index) => {
            // Determine if this is the currently streaming message
            const isCurrentlyStreaming = isStreaming && index === messages.length - 1;

            return (
              <ChatMessage
                key={index}
                messageIndex={index}
                role={message.role}
                content={message.content}
                reasoningEvents={message.reasoningEvents}
                isStreaming={isCurrentlyStreaming}
                usage={message.usage}
                hasSequenceNumber={!!message.sequenceNumber}
                onDelete={onDeleteMessage}
                onRegenerate={onRegenerateMessage}
                onBranch={onBranchConversation}
              />
            );
          })}

          {/* Streaming indicator (when streaming but no content yet) */}
          {isStreaming && messages.length > 0 && !messages[messages.length - 1].content && (
            <div className="pl-16">
              <StreamingIndicator />
            </div>
          )}
        </div>
      </div>
    );
  }
);

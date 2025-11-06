/**
 * ChatMessage component - renders a single chat message.
 *
 * Handles user and assistant messages with role-based styling.
 * Displays reasoning events via ReasoningAccordion.
 */

import * as React from 'react';
import { cn } from '../../lib/utils';
import type { ReasoningEvent } from '../../types/openai';
import { ReasoningAccordion } from './ReasoningAccordion';

export interface ChatMessageProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
  isStreaming?: boolean;
  className?: string;
}

/**
 * Renders a single chat message with role indicator and content.
 *
 * @example
 * ```tsx
 * <ChatMessage
 *   role="user"
 *   content="Hello!"
 * />
 *
 * <ChatMessage
 *   role="assistant"
 *   content="Hi there!"
 *   reasoningEvents={[...]}
 * />
 * ```
 */
export const ChatMessage = React.memo<ChatMessageProps>(
  ({ role, content, reasoningEvents, isStreaming = false, className }) => {
    return (
      <div className={cn('group', className)}>
        {/* Message bubble */}
        <div
          className={cn(
            'flex gap-4 rounded-lg p-4',
            role === 'user'
              ? 'bg-muted/50'
              : 'bg-background hover:bg-muted/30 transition-colors',
          )}
        >
          {/* Role indicator */}
          <div className="flex-shrink-0">
            <div
              className={cn(
                'flex h-8 w-8 items-center justify-center rounded-full text-xs font-medium',
                role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : role === 'system'
                    ? 'bg-muted text-muted-foreground'
                    : 'bg-secondary text-secondary-foreground',
              )}
            >
              {role === 'user' ? 'U' : role === 'system' ? 'S' : 'AI'}
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 space-y-3">
            {/* Reasoning events */}
            {reasoningEvents && reasoningEvents.length > 0 && (
              <ReasoningAccordion events={reasoningEvents} />
            )}

            {/* Message text */}
            <div className="prose prose-sm max-w-none overflow-hidden">
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground m-0 break-words">
                {content}
              </p>
            </div>

            {/* Streaming cursor */}
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-foreground ml-1 animate-pulse" />
            )}
          </div>
        </div>
      </div>
    );
  },
);

ChatMessage.displayName = 'ChatMessage';

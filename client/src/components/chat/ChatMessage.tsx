/**
 * ChatMessage component - renders a single chat message.
 *
 * Handles user and assistant messages with role-based styling.
 * Displays reasoning events via ReasoningAccordion.
 * Shows usage metadata and action buttons for assistant messages.
 */

import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { RefreshCw, GitBranch, Copy, Trash2 } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { ReasoningEvent, Usage } from '../../types/openai';
import { ReasoningAccordion } from './ReasoningAccordion';
import { Button } from '../ui/button';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '../ui/tooltip';

export interface ChatMessageProps {
  messageIndex: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
  isStreaming?: boolean;
  usage?: Usage | null;
  className?: string;
  hasSequenceNumber?: boolean;  // Whether this message is saved (has sequence number)
  onDelete?: (messageIndex: number) => void;
  onRegenerate?: (messageIndex: number) => void;
  onBranch?: (messageIndex: number) => void;
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
  ({
    messageIndex,
    role,
    content,
    reasoningEvents,
    isStreaming = false,
    usage,
    className,
    hasSequenceNumber = false,
    onDelete,
    onRegenerate,
    onBranch,
  }) => {
    const [clickedButton, setClickedButton] = React.useState<string | null>(null);

    const handleButtonClick = (buttonId: string, action: () => void) => {
      // Visual feedback
      setClickedButton(buttonId);
      setTimeout(() => setClickedButton(null), 200);

      // Execute action
      action();
    };

    const handleCopy = () => {
      navigator.clipboard.writeText(content);
    };

    return (
      <TooltipProvider>
        <div className={cn('group', className)}>
          {/* Message bubble */}
        <div
          className={cn(
            'flex gap-4 rounded-lg p-4',
            role === 'user'
              ? 'bg-muted/70'
              : 'bg-background',
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
            <div className="prose prose-sm max-w-none overflow-hidden prose-p:text-foreground prose-p:text-sm prose-p:leading-relaxed prose-p:m-0 prose-headings:text-foreground prose-strong:text-foreground prose-code:text-foreground prose-pre:bg-muted/50 prose-pre:text-foreground">
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                {content}
              </ReactMarkdown>
            </div>

            {/* Streaming cursor */}
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-foreground ml-1 animate-pulse" />
            )}

            {/* Message footer - metadata and actions */}
            {!isStreaming && (
              <div className="flex items-center justify-between pt-2 border-t border-muted/50">
                {/* Metadata (cost) - only for assistant messages */}
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  {role === 'assistant' && usage?.total_cost !== undefined ? (
                    <span title={`Total cost: $${usage.total_cost.toFixed(6)} (prompt: $${usage.prompt_cost?.toFixed(6) || '0.000000'} + completion: $${usage.completion_cost?.toFixed(6) || '0.000000'})`}>
                      ${usage.total_cost.toFixed(6)}
                    </span>
                  ) : null}
                </div>

                {/* Action buttons - always visible */}
                <div className="flex items-center gap-1">
                  {/* Copy button - always available */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className={cn(
                          'h-7 w-7 transition-colors hover:!bg-blue-100 dark:hover:!bg-blue-900/30',
                          clickedButton === 'copy' && 'bg-blue-100 dark:bg-blue-900/30'
                        )}
                        onClick={() => handleButtonClick('copy', handleCopy)}
                      >
                        <Copy className="h-3.5 w-3.5" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Copy message</TooltipContent>
                  </Tooltip>

                  {/* Assistant-only buttons */}
                  {role === 'assistant' && hasSequenceNumber && (
                    <>
                      {/* Branch button */}
                      {onBranch && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className={cn(
                                'h-7 w-7 transition-colors hover:!bg-blue-100 dark:hover:!bg-blue-900/30',
                                clickedButton === 'branch' && 'bg-blue-100 dark:bg-blue-900/30'
                              )}
                              onClick={() => handleButtonClick('branch', () => onBranch(messageIndex))}
                            >
                              <GitBranch className="h-3.5 w-3.5" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Branch conversation from this message</TooltipContent>
                        </Tooltip>
                      )}

                      {/* Regenerate button - rightmost of assistant buttons */}
                      {onRegenerate && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className={cn(
                                'h-7 w-7 transition-colors hover:!bg-blue-100 dark:hover:!bg-blue-900/30',
                                clickedButton === 'regenerate' && 'bg-blue-100 dark:bg-blue-900/30'
                              )}
                              onClick={() => handleButtonClick('regenerate', () => onRegenerate(messageIndex))}
                            >
                              <RefreshCw className="h-3.5 w-3.5" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Regenerate response</TooltipContent>
                        </Tooltip>
                      )}
                    </>
                  )}

                  {/* Delete button - only for user messages */}
                  {role === 'user' && hasSequenceNumber && onDelete && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className={cn(
                            'h-7 w-7 transition-colors hover:!bg-red-100 hover:!text-red-700 dark:hover:!bg-red-900/30 dark:hover:!text-red-400',
                            clickedButton === 'delete' && 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                          )}
                          onClick={() => handleButtonClick('delete', () => onDelete(messageIndex))}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Delete this message and all replies</TooltipContent>
                    </Tooltip>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      </TooltipProvider>
    );
  },
);

ChatMessage.displayName = 'ChatMessage';

/**
 * Minimal chat layout component.
 * Clean, ChatGPT-inspired interface for streaming conversations.
 */

import { useState, useRef, useEffect } from 'react';
import { Send, StopCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Accordion, AccordionItem } from './ui/accordion';
import { cn } from '../lib/utils';
import type { ReasoningEvent } from '../types/openai';
import { ReasoningEventType } from '../types/openai';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  reasoningEvents?: ReasoningEvent[];
}

export interface ChatLayoutProps {
  messages: Message[];
  isStreaming: boolean;
  onSendMessage: (content: string) => void;
  onCancel: () => void;
}

export function ChatLayout({
  messages,
  isStreaming,
  onSendMessage,
  onCancel,
}: ChatLayoutProps): JSX.Element {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    onSendMessage(input);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Messages area */}
      <ScrollArea ref={scrollRef} className="flex-1 px-4">
        <div className="mx-auto max-w-3xl py-8">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-semibold text-foreground mb-2">
                  Reasoning Agent
                </h2>
                <p className="text-muted-foreground">
                  Send a message to start a conversation
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <div key={index} className="group">
                  {/* Message bubble */}
                  <div
                    className={cn(
                      'flex gap-4 rounded-lg p-4',
                      message.role === 'user'
                        ? 'bg-muted/50'
                        : 'bg-background hover:bg-muted/30 transition-colors',
                    )}
                  >
                    {/* Role indicator */}
                    <div className="flex-shrink-0">
                      <div
                        className={cn(
                          'flex h-8 w-8 items-center justify-center rounded-full text-xs font-medium',
                          message.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-secondary text-secondary-foreground',
                        )}
                      >
                        {message.role === 'user' ? 'U' : 'AI'}
                      </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 space-y-3">
                      {/* Reasoning events */}
                      {message.reasoningEvents && message.reasoningEvents.length > 0 && (
                        <Accordion className="text-xs">
                          {message.reasoningEvents.map((event, i) => {
                            const hasMetadata = Object.keys(event.metadata).length > 0;
                            const showStep = event.type !== ReasoningEventType.ReasoningComplete;

                            return (
                              <AccordionItem
                                key={i}
                                title={
                                  <div className="flex items-center gap-2">
                                    <span
                                      className={cn(
                                        'inline-block h-1.5 w-1.5 rounded-full',
                                        (event.type === ReasoningEventType.IterationStart || event.type === ReasoningEventType.ToolExecutionStart) && 'bg-reasoning-search',
                                        event.type === ReasoningEventType.Planning && 'bg-reasoning-thinking',
                                        event.type === ReasoningEventType.ToolResult && 'bg-reasoning-action',
                                        (event.type === ReasoningEventType.IterationComplete || event.type === ReasoningEventType.ReasoningComplete) && 'bg-reasoning-complete',
                                        event.type === ReasoningEventType.Error && 'bg-red-500',
                                      )}
                                    />
                                    <span className="capitalize">
                                      {event.type.replace(/_/g, ' ')}
                                    </span>
                                    {showStep && (
                                      <span className="text-muted-foreground">
                                        Step {event.step_iteration}
                                      </span>
                                    )}
                                    {event.error && (
                                      <span className="text-red-500 ml-auto">Error</span>
                                    )}
                                  </div>
                                }
                              >
                                <div className="space-y-2">
                                  {event.error && (
                                    <div className="text-red-600">
                                      <span className="font-semibold">Error:</span> {event.error}
                                    </div>
                                  )}
                                  {hasMetadata && (
                                    <div className="font-mono text-xs bg-muted/50 rounded p-2 overflow-x-auto">
                                      <pre>{JSON.stringify(event.metadata, null, 2)}</pre>
                                    </div>
                                  )}
                                  {!hasMetadata && !event.error && (
                                    <div className="text-muted-foreground italic">No details</div>
                                  )}
                                </div>
                              </AccordionItem>
                            );
                          })}
                        </Accordion>
                      )}

                      {/* Message text */}
                      <div className="prose prose-sm max-w-none">
                        <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground m-0">
                          {message.content}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Streaming indicator */}
              {isStreaming && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground pl-16">
                  <div className="flex gap-1">
                    <span className="animate-pulse">●</span>
                    <span className="animate-pulse delay-75">●</span>
                    <span className="animate-pulse delay-150">●</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input area */}
      <div className="border-t bg-background">
        <div className="mx-auto max-w-3xl p-4">
          <form onSubmit={handleSubmit} className="relative">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              disabled={isStreaming}
              className="min-h-[60px] resize-none pr-12"
            />
            <div className="absolute bottom-2 right-2 flex gap-2">
              {isStreaming ? (
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  onClick={onCancel}
                  className="h-8 w-8"
                >
                  <StopCircle className="h-4 w-4" />
                </Button>
              ) : (
                <Button
                  type="submit"
                  size="icon"
                  disabled={!input.trim()}
                  className="h-8 w-8"
                >
                  <Send className="h-4 w-4" />
                </Button>
              )}
            </div>
          </form>
          <p className="mt-2 text-xs text-center text-muted-foreground">
            Press Enter to send, Shift + Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}

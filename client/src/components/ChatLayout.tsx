/**
 * ChatLayout component - main chat interface layout.
 *
 * Handles layout structure, auto-scrolling, and input form.
 * Delegates message rendering to ChatMessage component.
 */

import { useState, useRef, useEffect } from 'react';
import { Send, StopCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { ChatMessage } from './chat/ChatMessage';
import { StreamingIndicator } from './chat/StreamingIndicator';
import type { ReasoningEvent } from '../types/openai';

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
              {messages.map((message, index) => {
                // Determine if this is the currently streaming message
                const isCurrentlyStreaming = isStreaming && index === messages.length - 1;

                return (
                  <ChatMessage
                    key={index}
                    role={message.role}
                    content={message.content}
                    reasoningEvents={message.reasoningEvents}
                    isStreaming={isCurrentlyStreaming}
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

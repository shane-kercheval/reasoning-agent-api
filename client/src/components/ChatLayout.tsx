/**
 * ChatLayout component - main chat interface layout.
 *
 * Handles layout structure and auto-scrolling.
 * Delegates message rendering to ChatMessage component.
 */

import * as React from 'react';
import { useRef, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { ScrollArea } from './ui/scroll-area';
import { ChatMessage } from './chat/ChatMessage';
import { StreamingIndicator } from './chat/StreamingIndicator';
import { MessageInput, type MessageInputRef } from './forms/MessageInput';
import type { ReasoningEvent, Usage } from '../types/openai';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningEvents?: ReasoningEvent[];
  usage?: Usage | null;
}

export interface ChatLayoutProps {
  messages: Message[];
  isStreaming: boolean;
  isLoadingHistory?: boolean;
  input: string;
  onInputChange: (value: string) => void;
  onSendMessage: (content: string) => void;
  onCancel: () => void;
}

export interface ChatLayoutRef {
  focusInput: () => void;
  messageInput: MessageInputRef | null;
}

export const ChatLayout = React.forwardRef<ChatLayoutRef, ChatLayoutProps>(
  function ChatLayout(
    {
      messages,
      isStreaming,
      isLoadingHistory = false,
      input,
      onInputChange,
      onSendMessage,
      onCancel,
    },
    ref
  ) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<MessageInputRef>(null);

  // Expose focus method and message input ref to parent
  React.useImperativeHandle(ref, () => ({
    focusInput: () => messageInputRef.current?.focus(),
    messageInput: messageInputRef.current,
  }));

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex h-full flex-col bg-background overflow-hidden">
      {/* Messages area */}
      <ScrollArea ref={scrollRef} className="flex-1 px-4">
        {isLoadingHistory ? (
          /* Loading conversation history */
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
              <p className="text-base font-medium text-foreground">Loading conversation...</p>
              <p className="text-sm text-muted-foreground mt-1">Please wait</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-foreground mb-2">
                Assistant
              </h2>
              <p className="text-muted-foreground">
                Send a message to start a conversation
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl py-8 overflow-x-hidden">
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
                    usage={message.usage}
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
        )}
      </ScrollArea>

      {/* Input area */}
      <div className="border-t bg-background">
        <div className="mx-auto max-w-3xl p-4">
          <MessageInput
            ref={messageInputRef}
            value={input}
            onChange={onInputChange}
            onSend={onSendMessage}
            onCancel={onCancel}
            isStreaming={isStreaming}
          />
        </div>
      </div>
    </div>
  );
});

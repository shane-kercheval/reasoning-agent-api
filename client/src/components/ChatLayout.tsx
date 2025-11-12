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
import { MessageList } from './chat/MessageList';
import { MessageInput, type MessageInputRef } from './forms/MessageInput';
import type { ReasoningEvent, Usage } from '../types/openai';

export interface Message {
  id?: string;  // UUID from database
  sequenceNumber?: number;  // Sequence number from database
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
  isSettingsOpen?: boolean;
  settingsPanel?: React.ReactNode;
  // Message action handlers (optional - only available for saved messages)
  onDeleteMessage?: (messageIndex: number) => void;
  onRegenerateMessage?: (messageIndex: number) => void;
  onBranchConversation?: (messageIndex: number) => void;
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
      isSettingsOpen = false,
      settingsPanel,
      onDeleteMessage,
      onRegenerateMessage,
      onBranchConversation,
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
    <div className="flex h-full bg-background overflow-hidden">
      {/* Main chat area (messages + input) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Messages area */}
        <ScrollArea ref={scrollRef} className="flex-1 px-4">
        {isLoadingHistory && messages.length === 0 ? (
          /* Loading conversation history (only show spinner when no messages yet) */
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
          <MessageList
            messages={messages}
            isStreaming={isStreaming}
            onDeleteMessage={onDeleteMessage}
            onRegenerateMessage={onRegenerateMessage}
            onBranchConversation={onBranchConversation}
          />
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

      {/* Settings Panel (right sidebar within chat pane) */}
      {isSettingsOpen && settingsPanel && (
        <div className="w-80 flex-shrink-0 overflow-hidden border-l">
          {settingsPanel}
        </div>
      )}
    </div>
  );
});

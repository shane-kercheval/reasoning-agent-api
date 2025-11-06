/**
 * ConversationList component - list of conversations in sidebar.
 *
 * Displays all conversations with search, new conversation button,
 * and loading/error states.
 */

import { Plus, RefreshCw } from 'lucide-react';
import { ConversationItem } from './ConversationItem';
import { Button } from '../ui/button';
import { ScrollArea } from '../ui/scroll-area';
import type { ConversationSummary } from '../../lib/api-client';

export interface ConversationListProps {
  conversations: ConversationSummary[];
  selectedConversationId: string | null;
  isLoading: boolean;
  error: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => Promise<void>;
  onUpdateTitle: (id: string, title: string | null) => Promise<void>;
  onRefresh: () => void;
}

/**
 * Sidebar conversation list with actions.
 *
 * @example
 * ```tsx
 * <ConversationList
 *   conversations={conversations}
 *   selectedConversationId={currentId}
 *   isLoading={loading}
 *   error={error}
 *   onSelectConversation={handleSelect}
 *   onNewConversation={handleNew}
 *   onDeleteConversation={handleDelete}
 *   onUpdateTitle={handleUpdate}
 *   onRefresh={handleRefresh}
 * />
 * ```
 */
export function ConversationList({
  conversations,
  selectedConversationId,
  isLoading,
  error,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onUpdateTitle,
  onRefresh,
}: ConversationListProps): JSX.Element {
  return (
    <div className="flex flex-col h-full bg-background border-r">
      {/* Header with New Conversation button */}
      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Conversations</h2>
          <Button
            size="icon"
            variant="ghost"
            onClick={onRefresh}
            disabled={isLoading}
            title="Refresh conversations"
            className="h-8 w-8"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        <Button
          onClick={onNewConversation}
          className="w-full"
          size="sm"
        >
          <Plus className="h-4 w-4 mr-2" />
          New Conversation
        </Button>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-destructive/10 text-destructive text-sm">
          <p className="font-medium">Error loading conversations</p>
          <p className="text-xs mt-1">{error}</p>
        </div>
      )}

      {/* Initial loading state (no conversations loaded yet) */}
      {isLoading && conversations.length === 0 ? (
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center">
            <RefreshCw className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
            <p className="text-base font-medium text-foreground">Loading conversations...</p>
            <p className="text-sm text-muted-foreground mt-1">Please wait</p>
          </div>
        </div>
      ) : conversations.length === 0 ? (
        /* Empty state */
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-center">
            <p className="text-sm text-muted-foreground">No conversations yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Start a new conversation to begin
            </p>
          </div>
        </div>
      ) : (
        /* Conversation list (with optional refresh overlay) */
        <div className="flex-1 relative">
          {/* Refresh overlay (shows when refreshing existing list) */}
          {isLoading && (
            <div className="absolute inset-0 bg-background/50 backdrop-blur-sm flex items-center justify-center z-10">
              <div className="text-center">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2 text-primary" />
                <p className="text-sm font-medium">Refreshing...</p>
              </div>
            </div>
          )}

          <ScrollArea className="h-full">
            <div className="p-2 space-y-1">
              {conversations.map((conversation) => (
                <ConversationItem
                  key={conversation.id}
                  conversation={conversation}
                  isSelected={selectedConversationId === conversation.id}
                  onClick={() => onSelectConversation(conversation.id)}
                  onDelete={onDeleteConversation}
                  onUpdateTitle={onUpdateTitle}
                />
              ))}
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );
}

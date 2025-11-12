/**
 * ConversationList component - list of conversations in sidebar.
 *
 * Displays all conversations with search and loading/error states.
 */

import * as React from 'react';
import { RefreshCw, Search } from 'lucide-react';
import { ConversationItem } from './ConversationItem';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import type { ConversationSummary } from '../../lib/api-client';
import type { ConversationViewFilter } from '../../store/conversations-store';

export interface ConversationListProps {
  conversations: ConversationSummary[];
  selectedConversationId: string | null;
  isLoading: boolean;
  error: string | null;
  viewFilter: ConversationViewFilter;
  searchQuery: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => Promise<void>;
  onArchiveConversation: (id: string) => Promise<void>;
  onUpdateTitle: (id: string, title: string | null) => Promise<void>;
  onRefresh: () => void;
  onViewFilterChange: (filter: ConversationViewFilter) => void;
  onSearchQueryChange: (query: string) => void;
  onSearch: (query: string) => void;
}

/**
 * Sidebar conversation list with actions.
 *
 * @example
 * ```tsx
 * const listRef = useRef<ConversationListRef>(null);
 * <ConversationList
 *   ref={listRef}
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
export interface ConversationListRef {
  focusSearch: () => void;
  searchInput: HTMLInputElement | null;
}

export const ConversationList = React.forwardRef<ConversationListRef, ConversationListProps>(
  function ConversationList(
    {
      conversations,
      selectedConversationId,
      isLoading,
      error,
      viewFilter,
      searchQuery,
      onSelectConversation,
      onNewConversation: _onNewConversation,
      onDeleteConversation,
      onArchiveConversation,
      onUpdateTitle,
      onRefresh,
      onViewFilterChange,
      onSearchQueryChange,
      onSearch,
    },
    ref
  ) {
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  // Expose focus method and search input ref to parent
  React.useImperativeHandle(ref, () => ({
    focusSearch: () => searchInputRef.current?.focus(),
    searchInput: searchInputRef.current,
  }));

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && searchQuery.trim()) {
      onSearch(searchQuery.trim());
    }
  };
  return (
    <div className="flex flex-col h-full bg-background border-r">
      {/* Header */}
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

        {/* Search bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          <Input
            ref={searchInputRef}
            placeholder="Search messages... (press Enter)"
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            className="pl-9 h-9"
          />
        </div>

        {/* View filter toggle */}
        <div className="flex gap-1 p-1 bg-muted rounded-lg">
          <Button
            size="sm"
            variant={viewFilter === 'active' ? 'default' : 'ghost'}
            onClick={() => onViewFilterChange('active')}
            className="flex-1 h-7 text-xs"
          >
            Active
          </Button>
          <Button
            size="sm"
            variant={viewFilter === 'archived' ? 'default' : 'ghost'}
            onClick={() => onViewFilterChange('archived')}
            className="flex-1 h-7 text-xs"
          >
            Archived
          </Button>
        </div>
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
        <div className="flex-1 relative overflow-auto">
          {/* Refresh overlay (shows when refreshing existing list) */}
          {isLoading && (
            <div className="absolute inset-0 bg-background/50 backdrop-blur-sm flex items-center justify-center z-10">
              <div className="text-center">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2 text-primary" />
                <p className="text-sm font-medium">Refreshing...</p>
              </div>
            </div>
          )}

          <div className="p-2 space-y-1">
            {conversations.map((conversation) => (
              <ConversationItem
                key={conversation.id}
                conversation={conversation}
                isSelected={selectedConversationId === conversation.id}
                onClick={() => onSelectConversation(conversation.id)}
                onDelete={onDeleteConversation}
                onArchive={onArchiveConversation}
                onUpdateTitle={onUpdateTitle}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

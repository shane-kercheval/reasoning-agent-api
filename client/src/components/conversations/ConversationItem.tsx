/**
 * ConversationItem component - single conversation in the list.
 *
 * Displays conversation title, timestamp, and provides edit/delete actions.
 * Shows first N characters of first message if no title exists.
 */

import { useState } from 'react';
import { MessageSquare, Trash2, Edit2, Check, X } from 'lucide-react';
import type { ConversationSummary } from '../../lib/api-client';
import { Button } from '../ui/button';

export interface ConversationItemProps {
  conversation: ConversationSummary;
  isSelected: boolean;
  onClick: () => void;
  onDelete: (id: string) => Promise<void>;
  onUpdateTitle: (id: string, title: string | null) => Promise<void>;
}

/**
 * Single conversation item in the sidebar list.
 *
 * @example
 * ```tsx
 * <ConversationItem
 *   conversation={conv}
 *   isSelected={selectedId === conv.id}
 *   onClick={() => handleSelect(conv.id)}
 *   onDelete={handleDelete}
 *   onUpdateTitle={handleUpdateTitle}
 * />
 * ```
 */
export function ConversationItem({
  conversation,
  isSelected,
  onClick,
  onDelete,
  onUpdateTitle,
}: ConversationItemProps): JSX.Element {
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(conversation.title || '');
  const [isDeleting, setIsDeleting] = useState(false);

  // Generate display title
  const displayTitle = conversation.title || 'Untitled';

  // Format timestamp with date and time
  const formattedDateTime = new Date(conversation.updated_at).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });

  const handleSaveEdit = async () => {
    try {
      const newTitle = editTitle.trim() || null;
      await onUpdateTitle(conversation.id, newTitle);
      setIsEditing(false);
    } catch (err) {
      // Error handling done by parent
      console.error('Failed to update title:', err);
    }
  };

  const handleCancelEdit = () => {
    setEditTitle(conversation.title || '');
    setIsEditing(false);
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await onDelete(conversation.id);
    } catch (err) {
      setIsDeleting(false);
      // Error handling done by parent
      console.error('Failed to delete conversation:', err);
    }
  };

  return (
    <div
      className={`
        group relative p-3 rounded-lg cursor-pointer
        transition-colors duration-150
        ${isSelected ? 'bg-primary/10 border-l-2 border-primary' : 'hover:bg-muted/50'}
        ${isDeleting ? 'opacity-50 pointer-events-none' : ''}
      `}
      onClick={() => !isEditing && onClick()}
    >
      {/* Editing mode */}
      {isEditing ? (
        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          <input
            type="text"
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSaveEdit();
              if (e.key === 'Escape') handleCancelEdit();
            }}
            className="flex-1 px-2 py-1 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-primary"
            autoFocus
            placeholder="Conversation title"
          />
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6"
            onClick={handleSaveEdit}
            title="Save"
          >
            <Check className="h-3 w-3" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6"
            onClick={handleCancelEdit}
            title="Cancel"
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
      ) : (
        <>
          {/* Normal mode */}
          <div className="flex items-start gap-2">
            <MessageSquare className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium truncate">{displayTitle}</div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {formattedDateTime} • {conversation.message_count} messages
              </div>
            </div>
          </div>

          {/* Action buttons (show on hover) */}
          <div
            className="absolute right-2 top-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => e.stopPropagation()}
          >
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={() => setIsEditing(true)}
              title="Edit title"
            >
              <Edit2 className="h-3 w-3" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6 hover:bg-destructive hover:text-destructive-foreground"
              onClick={handleDelete}
              title="Delete conversation"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

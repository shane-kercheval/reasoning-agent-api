/**
 * MessageInput component - chat message input form.
 *
 * Handles message input with send/cancel buttons and keyboard shortcuts.
 */

import * as React from 'react';
import { Send, StopCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';

export interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: (message: string) => void;
  onCancel?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

/**
 * Message input form with send/cancel functionality.
 *
 * @example
 * ```tsx
 * const [input, setInput] = useState('');
 * <MessageInput
 *   value={input}
 *   onChange={setInput}
 *   onSend={(msg) => sendMessage(msg)}
 *   isStreaming={isStreaming}
 * />
 * ```
 */
export function MessageInput({
  value,
  onChange,
  onSend,
  onCancel,
  isStreaming = false,
  disabled = false,
  placeholder = 'Send a message...',
  className,
}: MessageInputProps): JSX.Element {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || isStreaming || disabled) return;
    onSend(value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={className}>
      <form onSubmit={handleSubmit} className="relative">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isStreaming || disabled}
          className="min-h-[60px] resize-none pr-12"
        />
        <div className="absolute bottom-2 right-2 flex gap-2">
          {isStreaming && onCancel ? (
            <Button
              type="button"
              size="icon"
              variant="ghost"
              onClick={onCancel}
              className="h-8 w-8"
              title="Stop generating"
            >
              <StopCircle className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              disabled={!value.trim() || disabled}
              className="h-8 w-8"
              title="Send message (Enter)"
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
  );
}

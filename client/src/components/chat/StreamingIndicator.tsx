/**
 * StreamingIndicator component - shows streaming state.
 *
 * Displays animated dots to indicate active streaming.
 */

import * as React from 'react';
import { cn } from '../../lib/utils';

export interface StreamingIndicatorProps {
  className?: string;
}

/**
 * Animated indicator showing streaming is in progress.
 *
 * @example
 * ```tsx
 * {isStreaming && <StreamingIndicator />}
 * ```
 */
export const StreamingIndicator = React.memo<StreamingIndicatorProps>(({ className }) => {
  return (
    <div className={cn('flex items-center gap-2 text-sm text-muted-foreground', className)}>
      <div className="flex gap-1">
        <span className="animate-pulse">●</span>
        <span className="animate-pulse delay-75" style={{ animationDelay: '75ms' }}>
          ●
        </span>
        <span className="animate-pulse delay-150" style={{ animationDelay: '150ms' }}>
          ●
        </span>
      </div>
    </div>
  );
});

StreamingIndicator.displayName = 'StreamingIndicator';

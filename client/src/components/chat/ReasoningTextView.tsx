/**
 * ReasoningTextView component - natural language display of reasoning events.
 *
 * Displays reasoning events as plain text without nested components or JSON details.
 * Provides a more user-friendly, scannable view of the AI's thinking process.
 */

import * as React from 'react';
import type { ReasoningEvent } from '../../types/openai';
import {
  groupEventsByIteration,
  formatIterationAsText,
} from '../../lib/reasoning-text-formatter';
import { cn } from '../../lib/utils';

export interface ReasoningTextViewProps {
  events: ReasoningEvent[];
  className?: string;
}

/**
 * Renders reasoning events as natural language text.
 *
 * Events are grouped by iteration and displayed as plain text blocks.
 * Structural events (ITERATION_START, ITERATION_COMPLETE, REASONING_COMPLETE)
 * are hidden to focus on meaningful content.
 *
 * @example
 * ```tsx
 * <ReasoningTextView
 *   events={[
 *     { type: 'planning', metadata: { thought: 'Need to search...' } },
 *     { type: 'tool_execution_start', metadata: { tools: ['web_search'] } }
 *   ]}
 * />
 * ```
 */
export const ReasoningTextView = React.memo<ReasoningTextViewProps>(
  ({ events, className }) => {
    // Group events by iteration for better organization
    const iterations = React.useMemo(
      () => groupEventsByIteration(events),
      [events]
    );

    if (iterations.length === 0) {
      return null;
    }

    return (
      <div className={cn('space-y-4', className)}>
        {iterations.map((iteration, idx) => {
          const text = formatIterationAsText(iteration);

          // Skip iterations with no displayable text
          if (!text) return null;

          return (
            <div key={idx}>
              {/* Natural language text content */}
              <div className="text-xs leading-relaxed whitespace-pre-wrap text-foreground/90 italic">
                {text}
              </div>
            </div>
          );
        })}
      </div>
    );
  }
);

ReasoningTextView.displayName = 'ReasoningTextView';

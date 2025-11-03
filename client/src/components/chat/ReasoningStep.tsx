/**
 * ReasoningStep component - renders a single reasoning event.
 *
 * Displays event type, step number, and metadata with type-specific
 * icons and colors for visual differentiation.
 */

import * as React from 'react';
import {
  Search,
  Brain,
  Wrench,
  CheckCircle,
  PlayCircle,
  AlertCircle,
  LucideIcon,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { ReasoningEventType, type ReasoningEvent } from '../../types/openai';

/**
 * Maps reasoning event types to their visual representation.
 */
const EVENT_CONFIG: Record<
  ReasoningEventType,
  {
    icon: LucideIcon;
    color: string;
    dotColor: string;
    label: string;
  }
> = {
  [ReasoningEventType.IterationStart]: {
    icon: PlayCircle,
    color: 'text-reasoning-search',
    dotColor: 'bg-reasoning-search',
    label: 'Iteration Start',
  },
  [ReasoningEventType.Planning]: {
    icon: Brain,
    color: 'text-reasoning-thinking',
    dotColor: 'bg-reasoning-thinking',
    label: 'Planning',
  },
  [ReasoningEventType.ToolExecutionStart]: {
    icon: Search,
    color: 'text-reasoning-search',
    dotColor: 'bg-reasoning-search',
    label: 'Tool Execution Start',
  },
  [ReasoningEventType.ToolResult]: {
    icon: Wrench,
    color: 'text-reasoning-action',
    dotColor: 'bg-reasoning-action',
    label: 'Tool Result',
  },
  [ReasoningEventType.IterationComplete]: {
    icon: CheckCircle,
    color: 'text-reasoning-complete',
    dotColor: 'bg-reasoning-complete',
    label: 'Iteration Complete',
  },
  [ReasoningEventType.ReasoningComplete]: {
    icon: CheckCircle,
    color: 'text-reasoning-complete',
    dotColor: 'bg-reasoning-complete',
    label: 'Reasoning Complete',
  },
  [ReasoningEventType.Error]: {
    icon: AlertCircle,
    color: 'text-red-500',
    dotColor: 'bg-red-500',
    label: 'Error',
  },
};

export interface ReasoningStepProps {
  event: ReasoningEvent;
  /** Whether to show the step number (hidden for ReasoningComplete) */
  showStep?: boolean;
  className?: string;
}

/**
 * Renders a single reasoning event with icon, label, and metadata.
 *
 * @example
 * ```tsx
 * <ReasoningStep
 *   event={{
 *     type: ReasoningEventType.Planning,
 *     step_iteration: 1,
 *     metadata: { plan: "Check weather" }
 *   }}
 * />
 * ```
 */
export const ReasoningStep = React.memo<ReasoningStepProps>(
  ({ event, showStep = true, className }) => {
    const config = EVENT_CONFIG[event.type];
    const Icon = config.icon;

    return (
      <div className={cn('flex items-center gap-2 text-xs', className)}>
        {/* Icon */}
        <Icon className={cn('h-3.5 w-3.5 shrink-0', config.color)} />

        {/* Label */}
        <span className="font-medium">{config.label}</span>

        {/* Step number */}
        {showStep && (
          <span className="text-muted-foreground">Step {event.step_iteration}</span>
        )}

        {/* Error indicator */}
        {event.error && <span className="text-red-500 ml-auto">Error</span>}
      </div>
    );
  },
);

ReasoningStep.displayName = 'ReasoningStep';

/**
 * Renders the metadata content for a reasoning event.
 */
export const ReasoningStepMetadata = React.memo<{ event: ReasoningEvent }>(({ event }) => {
  const hasMetadata = Object.keys(event.metadata).length > 0;

  return (
    <div className="space-y-2">
      {/* Error message */}
      {event.error && (
        <div className="text-red-600">
          <span className="font-semibold">Error:</span> {event.error}
        </div>
      )}

      {/* Metadata */}
      {hasMetadata && (
        <div className="font-mono text-xs bg-muted/50 rounded p-2 overflow-x-auto max-w-full">
          <pre className="whitespace-pre-wrap break-words m-0">
            {JSON.stringify(event.metadata, null, 2)}
          </pre>
        </div>
      )}

      {/* Empty state */}
      {!hasMetadata && !event.error && (
        <div className="text-muted-foreground italic">No details</div>
      )}
    </div>
  );
});

ReasoningStepMetadata.displayName = 'ReasoningStepMetadata';

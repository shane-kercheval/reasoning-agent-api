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
 * Renders a value based on its type with appropriate formatting.
 */
const renderValue = (value: unknown, depth = 0): React.ReactNode => {
  // Handle null/undefined
  if (value === null || value === undefined) {
    return <span className="text-muted-foreground italic">None</span>;
  }

  // Handle boolean
  if (typeof value === 'boolean') {
    return (
      <span className="font-medium">
        {value ? '✓ Yes' : '✗ No'}
      </span>
    );
  }

  // Handle number
  if (typeof value === 'number') {
    return <span className="font-medium">{value}</span>;
  }

  // Handle string
  if (typeof value === 'string') {
    // Empty string
    if (value.trim() === '') {
      return <span className="text-muted-foreground italic">Empty</span>;
    }
    // Regular string
    return <p className="text-sm leading-relaxed whitespace-pre-wrap">{value}</p>;
  }

  // Handle array
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-muted-foreground italic">None</span>;
    }
    return (
      <ul className="list-disc list-inside space-y-1 text-sm">
        {value.map((item, idx) => (
          <li key={idx}>{renderValue(item, depth + 1)}</li>
        ))}
      </ul>
    );
  }

  // Handle object (nested)
  if (typeof value === 'object') {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return <span className="text-muted-foreground italic">Empty</span>;
    }
    return (
      <div className={cn('space-y-2', depth > 0 && 'pl-4 border-l-2 border-muted')}>
        {entries.map(([key, val]) => (
          <div key={key}>
            <div className="font-medium text-sm text-muted-foreground capitalize">
              {key.replace(/_/g, ' ')}
            </div>
            <div className="mt-1">{renderValue(val, depth + 1)}</div>
          </div>
        ))}
      </div>
    );
  }

  // Fallback for unknown types
  return <span className="font-mono text-xs">{String(value)}</span>;
};

/**
 * Renders the metadata content for a reasoning event.
 */
export const ReasoningStepMetadata = React.memo<{ event: ReasoningEvent }>(({ event }) => {
  const hasMetadata = Object.keys(event.metadata).length > 0;

  return (
    <div className="space-y-3">
      {/* Error message */}
      {event.error && (
        <div className="text-red-600">
          <span className="font-semibold">Error:</span> {event.error}
        </div>
      )}

      {/* Metadata with smart rendering */}
      {hasMetadata && (
        <div className="space-y-3">
          {Object.entries(event.metadata).map(([key, value]) => (
            <div key={key}>
              <div className="font-semibold text-sm text-foreground/80 capitalize mb-1.5">
                {key.replace(/_/g, ' ')}:
              </div>
              <div className="pl-3">{renderValue(value)}</div>
            </div>
          ))}
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

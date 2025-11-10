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
  ChevronRight,
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
  [ReasoningEventType.ExternalReasoning]: {
    icon: Brain,
    color: 'text-reasoning-thinking',
    dotColor: 'bg-reasoning-thinking',
    label: 'Model Reasoning',
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
 * Checks if a value is empty and should not be displayed.
 * Returns true for null, undefined, empty strings, empty arrays, and empty objects.
 */
const isEmpty = (value: unknown): boolean => {
  if (value === null || value === undefined) {
    return true;
  }
  if (typeof value === 'string' && value.trim() === '') {
    return true;
  }
  if (Array.isArray(value) && value.length === 0) {
    return true;
  }
  if (typeof value === 'object' && Object.keys(value).length === 0) {
    return true;
  }
  return false;
};

/**
 * Filters an object to remove empty values (null, undefined, empty arrays, empty objects).
 */
const filterEmptyValues = (obj: Record<string, unknown>): Record<string, unknown> => {
  const filtered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (!isEmpty(value)) {
      filtered[key] = value;
    }
  }
  return filtered;
};

/**
 * Single collapsible item with chevron.
 */
const CollapsibleItem = ({
  value,
  depth,
  index
}: {
  value: unknown;
  depth: number;
  index: number;
}): JSX.Element => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  // For primitives, just show inline without collapse
  const isPrimitive = typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean';

  if (isPrimitive) {
    return (
      <div className="flex gap-2 items-start">
        <span className="text-muted-foreground text-xs mt-0.5">•</span>
        <div className="flex-1 min-w-0">{renderValue(value, depth + 1)}</div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex gap-1 items-start w-full text-left hover:bg-muted/30 rounded px-1 -ml-1 transition-colors"
      >
        <ChevronRight
          className={cn(
            'h-3.5 w-3.5 shrink-0 text-muted-foreground mt-0.5 transition-transform',
            isExpanded && 'rotate-90'
          )}
        />
        <span className="text-xs text-muted-foreground">Item {index + 1}</span>
      </button>
      {isExpanded && (
        <div className="pl-4 ml-1 border-l-2 border-muted/50">
          {renderValue(value, depth + 1)}
        </div>
      )}
    </div>
  );
};

/**
 * Collapsible array component with chevrons for each item.
 */
const CollapsibleArray = ({ items, depth }: { items: unknown[]; depth: number }): JSX.Element => {
  return (
    <div className="space-y-1.5">
      {items.map((item, idx) => (
        <CollapsibleItem key={idx} value={item} depth={depth} index={idx} />
      ))}
    </div>
  );
};

/**
 * Renders a value based on its type with appropriate formatting.
 */
const renderValue = (value: unknown, depth = 0): React.ReactNode => {
  // Handle null/undefined
  if (value === null || value === undefined) {
    return <span className="text-muted-foreground italic text-xs">None</span>;
  }

  // Handle boolean
  if (typeof value === 'boolean') {
    return (
      <span className="font-medium text-xs">
        {value ? '✓ Yes' : '✗ No'}
      </span>
    );
  }

  // Handle number
  if (typeof value === 'number') {
    return <span className="font-medium text-xs">{value}</span>;
  }

  // Handle string
  if (typeof value === 'string') {
    // Empty string
    if (value.trim() === '') {
      return <span className="text-muted-foreground italic text-xs">Empty</span>;
    }
    // Regular string
    return <p className="text-xs leading-relaxed whitespace-pre-wrap break-words">{value}</p>;
  }

  // Handle array
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-muted-foreground italic text-xs">None</span>;
    }
    return <CollapsibleArray items={value} depth={depth} />;
  }

  // Handle object (nested) - format as JSON code block
  if (typeof value === 'object') {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return <span className="text-muted-foreground italic text-xs">Empty</span>;
    }

    try {
      const jsonString = JSON.stringify(value, null, 2);
      return (
        <pre className="text-xs bg-muted/30 rounded p-2 border border-muted whitespace-pre-wrap break-words overflow-wrap-anywhere">
          <code className="text-foreground/90 font-mono">{jsonString}</code>
        </pre>
      );
    } catch (e) {
      // Fallback if JSON.stringify fails (circular refs, etc.)
      return <span className="font-mono text-xs break-all text-red-500">Invalid JSON object</span>;
    }
  }

  // Fallback for unknown types
  return <span className="font-mono text-xs break-all">{String(value)}</span>;
};

/**
 * Checks if a reasoning event has any displayable content.
 * Returns false if the event has no error and no non-empty metadata.
 */
export const hasDisplayableContent = (event: ReasoningEvent): boolean => {
  // Has error message
  if (event.error) {
    return true;
  }

  // Has non-empty metadata
  const filteredMetadata = filterEmptyValues(event.metadata);
  return Object.keys(filteredMetadata).length > 0;
};

/**
 * Renders the metadata content for a reasoning event.
 */
export const ReasoningStepMetadata = React.memo<{ event: ReasoningEvent }>(({ event }) => {
  // Filter out empty values from metadata
  const filteredMetadata = filterEmptyValues(event.metadata);
  const hasMetadata = Object.keys(filteredMetadata).length > 0;

  return (
    <div className="space-y-3">
      {/* Error message */}
      {event.error && (
        <div className="text-red-600">
          <span className="font-semibold">Error:</span> {event.error}
        </div>
      )}

      {/* Metadata with smart rendering - only show non-empty fields */}
      {hasMetadata && (
        <div className="space-y-3">
          {Object.entries(filteredMetadata).map(([key, value]) => (
            <div key={key}>
              <div className="font-semibold text-xs text-foreground/80 capitalize mb-1.5">
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

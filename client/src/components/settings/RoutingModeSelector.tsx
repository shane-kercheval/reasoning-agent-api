/**
 * RoutingModeSelector component - selector for API routing mode.
 *
 * Allows user to choose how requests are routed:
 * - Chat: Standard chat mode (fastest)
 * - Reasoning: Single-loop reasoning agent
 * - Auto: LLM classifier decides
 */

import { cn } from '../../lib/utils';
import { RoutingMode, type RoutingModeType } from '../../constants';

export interface RoutingModeSelectorProps {
  value: RoutingModeType;
  onChange: (mode: RoutingModeType) => void;
  className?: string;
}

const ROUTING_OPTIONS = [
  {
    value: RoutingMode.PASSTHROUGH,
    label: 'Chat',
    description: 'Standard chat mode. No tools. Fastest response.',
  },
  {
    value: RoutingMode.REASONING,
    label: 'Reasoning',
    description: 'Use sequential reasoning with tools.',
  },
  {
    value: RoutingMode.AUTO,
    label: 'Auto',
    description: 'Let AI decide',
  },
] as const;

/**
 * Segmented control for routing mode selection.
 *
 * @example
 * ```tsx
 * const [mode, setMode] = useState(RoutingMode.REASONING);
 * <RoutingModeSelector value={mode} onChange={setMode} />
 * ```
 */
export function RoutingModeSelector({
  value,
  onChange,
  className,
}: RoutingModeSelectorProps): JSX.Element {
  return (
    <div className={cn('space-y-2', className)}>
      <label className="text-sm font-medium text-foreground">Routing Mode</label>
      <div className="flex gap-1 p-1 bg-muted rounded-lg">
        {ROUTING_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={cn(
              'flex-1 px-3 py-2 text-xs font-medium rounded-md transition-colors',
              value === option.value
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
            title={option.description}
          >
            {option.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        {ROUTING_OPTIONS.find((opt) => opt.value === value)?.description}
      </p>
    </div>
  );
}

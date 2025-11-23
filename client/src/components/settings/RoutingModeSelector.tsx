/**
 * RoutingModeSelector component - selector for API routing mode.
 *
 * Allows user to choose how requests are routed:
 * - Chat: Standard chat mode (fastest)
 * - Reasoning: Single-loop reasoning agent
 * - Auto: LLM classifier decides
 */

import { RoutingMode, type RoutingModeType } from '../../constants';
import { SegmentedControl } from '../ui/segmented-control';

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
    <SegmentedControl
      label="Routing Mode"
      value={value}
      options={ROUTING_OPTIONS}
      onChange={onChange}
      className={className}
    />
  );
}

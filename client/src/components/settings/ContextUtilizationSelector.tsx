/**
 * ContextUtilizationSelector component - selector for context window utilization.
 *
 * Allows user to choose max percentage of context window to use:
 * - Low: 33% of context window
 * - Medium: 66% of context window
 * - Full: 100% of context window
 */

import { SegmentedControl } from '../ui/segmented-control';

export type ContextUtilizationType = 'low' | 'medium' | 'full';

export interface ContextUtilizationSelectorProps {
  value: ContextUtilizationType;
  onChange: (value: ContextUtilizationType) => void;
  className?: string;
}

const CONTEXT_UTILIZATION_OPTIONS = [
  {
    value: 'low' as const,
    label: 'Low',
    description: 'Use up to 33% of context window',
  },
  {
    value: 'medium' as const,
    label: 'Medium',
    description: 'Use up to 66% of context window',
  },
  {
    value: 'full' as const,
    label: 'Full',
    description: 'Use up to 100% of context window',
  },
] as const;

/**
 * Segmented control for context utilization selection.
 *
 * @example
 * ```tsx
 * const [utilization, setUtilization] = useState<ContextUtilizationType>('full');
 * <ContextUtilizationSelector value={utilization} onChange={setUtilization} />
 * ```
 */
export function ContextUtilizationSelector({
  value,
  onChange,
  className,
}: ContextUtilizationSelectorProps): JSX.Element {
  return (
    <SegmentedControl
      label="Context Utilization"
      value={value}
      options={CONTEXT_UTILIZATION_OPTIONS}
      onChange={onChange}
      className={className}
    />
  );
}

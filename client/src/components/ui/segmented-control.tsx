/**
 * SegmentedControl - Reusable segmented button control component.
 *
 * A pill-style selector for choosing between multiple options.
 * Used for settings like routing mode, context utilization, etc.
 */

import { cn } from '../../lib/utils';

export interface SegmentedControlOption<T extends string> {
  value: T;
  label: string;
  description: string;
}

export interface SegmentedControlProps<T extends string> {
  label: string;
  value: T;
  options: readonly SegmentedControlOption<T>[];
  onChange: (value: T) => void;
  className?: string;
}

/**
 * Generic segmented control for selecting between options.
 *
 * @example
 * ```tsx
 * const options = [
 *   { value: 'low', label: 'Low', description: 'Low usage (33%)' },
 *   { value: 'medium', label: 'Medium', description: 'Medium usage (66%)' },
 *   { value: 'full', label: 'Full', description: 'Full usage (100%)' },
 * ] as const;
 *
 * <SegmentedControl
 *   label="Context Utilization"
 *   value={value}
 *   options={options}
 *   onChange={setValue}
 * />
 * ```
 */
export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
  className,
}: SegmentedControlProps<T>): JSX.Element {
  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div className={cn('space-y-2', className)}>
      <label className="text-sm font-medium text-foreground">{label}</label>
      <div className="flex gap-1 p-1 bg-muted rounded-lg">
        {options.map((option) => (
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
      {selectedOption && (
        <p className="text-xs text-muted-foreground">
          {selectedOption.description}
        </p>
      )}
    </div>
  );
}

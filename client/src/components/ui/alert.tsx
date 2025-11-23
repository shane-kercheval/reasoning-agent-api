/**
 * Alert component for displaying inline notifications.
 *
 * Supports error, warning, and info variants with optional dismiss button.
 */

import { X } from 'lucide-react';

interface AlertProps {
  variant?: 'error' | 'warning' | 'info';
  message: string;
  onDismiss?: () => void;
}

export function Alert({ variant = 'error', message, onDismiss }: AlertProps): JSX.Element {
  const variantStyles = {
    error: 'bg-red-50 border-red-200 text-red-800 dark:bg-red-950 dark:border-red-800 dark:text-red-200',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-950 dark:border-yellow-800 dark:text-yellow-200',
    info: 'bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-200',
  };

  return (
    <div className={`border px-4 py-3 rounded-md flex items-start gap-3 ${variantStyles[variant]}`}>
      <div className="flex-1">
        <p className="text-sm font-medium">
          {variant === 'error' && 'Error: '}
          {message}
        </p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="flex-shrink-0 text-current opacity-70 hover:opacity-100 transition-opacity"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}

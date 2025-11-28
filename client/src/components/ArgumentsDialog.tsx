/**
 * ArgumentsDialog - Modal for collecting prompt/tool argument values.
 *
 * Shows a form with auto-expanding text areas for each argument.
 * Handles required vs optional arguments and validation.
 * Supports both prompts and tools with different button text and loading states.
 */

import { useEffect, useState, useRef } from 'react';
import { Card } from './ui/card';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { X, Loader2 } from 'lucide-react';
import type { PromptArgument, Tool } from '../lib/api-client';
import { convertArgumentsToTypes, getTypeLabel } from '../lib/schema-utils';

interface ArgumentsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  type: 'prompt' | 'tool';
  name: string;
  description?: string;
  arguments: PromptArgument[];
  inputSchema?: Tool['input_schema']; // For tools - provides type information
  onSubmit: (args: Record<string, unknown>) => void;
  isExecuting?: boolean;
}

export function ArgumentsDialog({
  isOpen,
  onClose,
  type,
  name,
  description,
  arguments: itemArgs,
  inputSchema,
  onSubmit,
  isExecuting = false,
}: ArgumentsDialogProps): JSX.Element | null {
  // State for argument values (all stored as strings in form, converted on submit)
  const [argValues, setArgValues] = useState<Record<string, string>>({});
  const firstInputRef = useRef<HTMLTextAreaElement>(null);

  // Reset form when dialog opens
  useEffect(() => {
    if (isOpen) {
      // Initialize with empty values
      const initialValues: Record<string, string> = {};
      itemArgs.forEach((arg) => {
        initialValues[arg.name] = '';
      });
      setArgValues(initialValues);

      // Focus first input
      setTimeout(() => firstInputRef.current?.focus(), 0);
    }
  }, [isOpen, itemArgs]);

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate required arguments
    const missingRequired = itemArgs
      .filter((arg) => arg.required && !argValues[arg.name]?.trim())
      .map((arg) => arg.name);

    if (missingRequired.length > 0) {
      alert(`Please fill in required fields: ${missingRequired.join(', ')}`);
      return;
    }

    try {
      // Convert string values to proper JSON types based on schema
      const typedArgs = convertArgumentsToTypes(argValues, inputSchema);
      onSubmit(typedArgs);
    } catch (error) {
      alert(
        `Invalid input: ${error instanceof Error ? error.message : 'Unknown error'}`,
      );
    }
  };

  // Handle input change
  const handleChange = (name: string, value: string) => {
    setArgValues((prev) => ({ ...prev, [name]: value }));
  };

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  // Button text based on type
  const buttonText = type === 'prompt' ? 'Insert Prompt' : 'Execute Tool';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <Card className="relative w-full max-w-2xl max-h-[80vh] overflow-y-auto p-6 shadow-2xl">
        {/* Header */}
        <div className="mb-4">
          <div className="flex items-start justify-between mb-2">
            <h2 className="text-lg font-semibold">{name}</h2>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Close"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          {description && (
            <div className="text-xs text-gray-500 dark:text-gray-400 whitespace-pre-line max-h-[250px] overflow-y-auto">
              {description}
            </div>
          )}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {itemArgs.map((arg, index) => {
            // Get type information from schema if available (for tools)
            const paramType = inputSchema?.properties?.[arg.name]?.type;
            const typeLabel = getTypeLabel(paramType);

            return (
              <div key={arg.name}>
                <label
                  htmlFor={`arg-${arg.name}`}
                  className="block text-sm font-medium mb-1.5"
                >
                  {arg.name}
                  {arg.required && <span className="text-red-500 ml-1">*</span>}
                  {paramType && (
                    <span className="ml-2 text-xs text-gray-500 dark:text-gray-400 font-normal">
                      ({typeLabel})
                    </span>
                  )}
                </label>
                {arg.description && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                    {arg.description}
                  </p>
                )}
                <Textarea
                  ref={index === 0 ? firstInputRef : undefined}
                  id={`arg-${arg.name}`}
                  value={argValues[arg.name] || ''}
                  onChange={(e) => handleChange(arg.name, e.target.value)}
                  placeholder={
                    arg.required
                      ? `Required (${typeLabel})`
                      : `Optional (${typeLabel})`
                  }
                  className="min-h-[60px] max-h-[200px] resize-none"
                  required={arg.required}
                  disabled={isExecuting}
                />
              </div>
            );
          })}

          {/* Footer buttons with help text */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">Esc</kbd>
              {' '}to cancel
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                onClick={onClose}
                disabled={isExecuting}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isExecuting}>
                {isExecuting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {isExecuting ? 'Executing...' : buttonText}
              </Button>
            </div>
          </div>
        </form>
      </Card>
    </div>
  );
}

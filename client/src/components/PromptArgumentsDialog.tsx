/**
 * PromptArgumentsDialog - Modal for collecting MCP prompt argument values.
 *
 * Shows a form with auto-expanding text areas for each prompt argument.
 * Handles required vs optional arguments and validation.
 */

import { useEffect, useState, useRef } from 'react';
import { Card } from './ui/card';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { X } from 'lucide-react';
import type { MCPPromptArgument } from '../lib/api-client';

interface PromptArgumentsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  promptName: string;
  promptDescription?: string;
  arguments: MCPPromptArgument[];
  onSubmit: (args: Record<string, string>) => void;
}

export function PromptArgumentsDialog({
  isOpen,
  onClose,
  promptName,
  promptDescription,
  arguments: promptArgs,
  onSubmit,
}: PromptArgumentsDialogProps): JSX.Element | null {
  // State for argument values
  const [argValues, setArgValues] = useState<Record<string, string>>({});
  const firstInputRef = useRef<HTMLTextAreaElement>(null);

  // Reset form when dialog opens
  useEffect(() => {
    if (isOpen) {
      // Initialize with empty values
      const initialValues: Record<string, string> = {};
      promptArgs.forEach((arg) => {
        initialValues[arg.name] = '';
      });
      setArgValues(initialValues);

      // Focus first input
      setTimeout(() => firstInputRef.current?.focus(), 0);
    }
  }, [isOpen, promptArgs]);

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate required arguments
    const missingRequired = promptArgs
      .filter((arg) => arg.required && !argValues[arg.name]?.trim())
      .map((arg) => arg.name);

    if (missingRequired.length > 0) {
      alert(`Please fill in required fields: ${missingRequired.join(', ')}`);
      return;
    }

    onSubmit(argValues);
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

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <Card className="relative w-full max-w-2xl max-h-[80vh] overflow-y-auto p-6 shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold">{promptName}</h2>
            {promptDescription && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {promptDescription}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Close"
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {promptArgs.map((arg, index) => (
            <div key={arg.name}>
              <label
                htmlFor={`arg-${arg.name}`}
                className="block text-sm font-medium mb-1.5"
              >
                {arg.name}
                {arg.required && <span className="text-red-500 ml-1">*</span>}
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
                placeholder={arg.required ? 'Required' : 'Optional'}
                className="min-h-[60px] max-h-[200px] resize-none"
                required={arg.required}
              />
            </div>
          ))}

          {/* Footer buttons */}
          <div className="flex items-center justify-end gap-2 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button
              type="button"
              variant="ghost"
              onClick={onClose}
            >
              Cancel
            </Button>
            <Button type="submit">
              Insert Prompt
            </Button>
          </div>
        </form>

        {/* Help text */}
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
          <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">Esc</kbd>
          {' '}to cancel
        </div>
      </Card>
    </div>
  );
}

/**
 * ToolExecutionDialog - Modal for displaying tool execution results.
 *
 * Shows the result of a tool execution with:
 * - Success/error indicator
 * - Execution time
 * - Result content (formatted JSON or text)
 * - Copy to clipboard button
 * - Send to chat button
 */

import { Card } from './ui/card';
import { Button } from './ui/button';
import { X, Copy, Send, CheckCircle, XCircle } from 'lucide-react';
import type { ToolExecutionResult } from '../lib/api-client';
import { formatToolResult } from '../lib/format-tool-result';

interface ToolExecutionDialogProps {
  isOpen: boolean;
  onClose: () => void;
  result: ToolExecutionResult | null;
  onCopy: () => void;
  onSendToChat: () => void;
}

export function ToolExecutionDialog({
  isOpen,
  onClose,
  result,
  onCopy,
  onSendToChat,
}: ToolExecutionDialogProps): JSX.Element | null {
  if (!isOpen || !result) return null;

  const formattedResult = formatToolResult(result.result);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <Card className="relative w-full max-w-3xl p-6 shadow-2xl flex flex-col max-h-[80vh]">
        {/* Header */}
        <div className="flex items-start justify-between mb-4 flex-shrink-0">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold">{result.tool_name}</h2>
              {result.success ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500" />
              )}
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Executed in {result.execution_time_ms.toFixed(2)}ms
            </p>
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

        {/* Result content */}
        <div className="mb-4 flex-1 min-h-0 flex flex-col">
          <h3 className="text-sm font-medium mb-2 flex-shrink-0">Result:</h3>
          <pre className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4 overflow-auto text-xs font-mono whitespace-pre-wrap break-words flex-1">
            {formattedResult}
          </pre>
        </div>

        {/* Action buttons with help text */}
        <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="text-xs text-gray-500 dark:text-gray-400">
            <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">Esc</kbd>
            {' '}to close
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              onClick={onCopy}
              className="gap-2"
            >
              <Copy className="h-4 w-4" />
              Copy
            </Button>
            <Button
              type="button"
              onClick={onSendToChat}
              className="gap-2"
            >
              <Send className="h-4 w-4" />
              Send to Chat
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}

/**
 * KeyboardShortcutsOverlay - Modal overlay displaying all keyboard shortcuts.
 *
 * Shows a comprehensive list of keyboard shortcuts organized by category.
 * Can be toggled with Shift+Cmd+/ (Shift+Ctrl+/ on Windows/Linux).
 */

import { useEffect } from 'react';
import { Card } from './ui/card';
import { X } from 'lucide-react';

interface KeyboardShortcutsOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
}

interface ShortcutSection {
  title: string;
  shortcuts: ShortcutItem[];
}

export function KeyboardShortcutsOverlay({ isOpen, onClose }: KeyboardShortcutsOverlayProps): JSX.Element | null {
  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const modKey = isMac ? 'Cmd' : 'Ctrl';

  const sections: ShortcutSection[] = [
    {
      title: 'General',
      shortcuts: [
        { keys: [modKey, 'N'], description: 'New conversation' },
        { keys: ['Shift', modKey, 'P'], description: 'Open command palette' },
        { keys: [modKey, ','], description: 'Toggle settings panel' },
        { keys: [modKey, '/'], description: 'Show keyboard shortcuts' },
        { keys: ['Escape'], description: 'Focus chat input' },
      ],
    },
    {
      title: 'Tabs',
      shortcuts: [
        { keys: [modKey, 'W'], description: 'Close current tab' },
        { keys: ['Shift', modKey, '['], description: 'Previous tab' },
        { keys: ['Shift', modKey, ']'], description: 'Next tab' },
      ],
    },
    {
      title: 'Navigation',
      shortcuts: [
        { keys: ['Shift', modKey, 'F'], description: 'Focus search box' },
      ],
    },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <Card className="relative w-full max-w-2xl max-h-[80vh] overflow-y-auto p-6 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-semibold">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="rounded-lg p-2 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Shortcuts sections */}
        <div className="space-y-6">
          {sections.map((section) => (
            <div key={section.title}>
              <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                {section.title}
              </h3>
              <div className="space-y-2">
                {section.shortcuts.map((shortcut, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <span className="text-sm">{shortcut.description}</span>
                    <div className="flex items-center gap-1">
                      {shortcut.keys.map((key, keyIndex) => (
                        <span key={keyIndex} className="flex items-center gap-1">
                          <kbd className="px-2 py-1 text-xs font-semibold bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded shadow-sm">
                            {key}
                          </kbd>
                          {keyIndex < shortcut.keys.length - 1 && (
                            <span className="text-gray-400 text-xs">+</span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer hint */}
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 text-center text-xs text-gray-500 dark:text-gray-400">
          Press <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">Escape</kbd> to close
        </div>
      </Card>
    </div>
  );
}

/**
 * CommandPalette - Modal for selecting and inserting MCP prompts.
 *
 * Keyboard shortcuts:
 * - Cmd+Shift+P (Ctrl+Shift+P on Windows/Linux) to open
 * - Type to filter prompts
 * - Up/Down arrows to navigate
 * - Enter to select
 * - Escape to close
 */

import { useEffect, useState, useRef } from 'react';
import { Card } from './ui/card';
import { Input } from './ui/input';
import { X, Search } from 'lucide-react';
import type { MCPPrompt } from '../lib/api-client';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  prompts: MCPPrompt[];
  onSelectPrompt: (prompt: MCPPrompt) => void;
  isLoading?: boolean;
}

export function CommandPalette({
  isOpen,
  onClose,
  prompts,
  onSelectPrompt,
  isLoading = false,
}: CommandPaletteProps): JSX.Element | null {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Filter prompts based on search query
  const filteredPrompts = prompts.filter((prompt) => {
    const query = searchQuery.toLowerCase();
    const nameMatch = prompt.name.toLowerCase().includes(query);
    const descMatch = prompt.description?.toLowerCase().includes(query) || false;
    return nameMatch || descMatch;
  });

  // Reset state when opened
  useEffect(() => {
    if (isOpen) {
      setSearchQuery('');
      setSelectedIndex(0);
      // Focus search input when opened
      setTimeout(() => searchInputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Reset selected index when filtered prompts change
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < filteredPrompts.length - 1 ? prev + 1 : prev
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredPrompts[selectedIndex]) {
            onSelectPrompt(filteredPrompts[selectedIndex]);
            onClose();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredPrompts, selectedIndex, onSelectPrompt, onClose]);

  // Scroll selected item into view
  useEffect(() => {
    if (!listRef.current) return;

    const selectedElement = listRef.current.children[selectedIndex] as HTMLElement;
    if (selectedElement) {
      selectedElement.scrollIntoView({
        block: 'nearest',
        behavior: 'smooth',
      });
    }
  }, [selectedIndex]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-black/50 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <Card className="relative w-full max-w-2xl shadow-2xl">
        {/* Header with search */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Command Palette</h2>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              ref={searchInputRef}
              type="text"
              placeholder="Search prompts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Prompts list */}
        <div
          ref={listRef}
          className="max-h-[400px] overflow-y-auto"
        >
          {isLoading ? (
            <div className="p-8 text-center text-gray-500">
              Loading prompts...
            </div>
          ) : filteredPrompts.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              {searchQuery ? 'No prompts found' : 'No prompts available'}
            </div>
          ) : (
            filteredPrompts.map((prompt, index) => (
              <button
                key={prompt.name}
                onClick={() => {
                  onSelectPrompt(prompt);
                  onClose();
                }}
                className={`w-full text-left px-4 py-3 border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors ${
                  index === selectedIndex
                    ? 'bg-blue-50 dark:bg-blue-900/20 border-l-2 border-l-blue-500'
                    : ''
                }`}
              >
                <div className="font-medium text-sm">{prompt.name}</div>
                {prompt.description && (
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {prompt.description}
                  </div>
                )}
                {prompt.arguments && prompt.arguments.length > 0 && (
                  <div className="text-xs text-gray-400 mt-1">
                    Arguments: {prompt.arguments.map(arg => arg.name).join(', ')}
                  </div>
                )}
              </button>
            ))
          )}
        </div>

        {/* Footer hint */}
        <div className="p-3 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 flex items-center justify-between">
          <div>
            <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">↑↓</kbd>
            {' '}to navigate
            {' '}
            <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">Enter</kbd>
            {' '}to select
          </div>
          <div>
            <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded">Esc</kbd>
            {' '}to close
          </div>
        </div>
      </Card>
    </div>
  );
}

/**
 * CommandPalette - Modal for selecting and inserting MCP prompts and tools.
 *
 * Keyboard shortcuts:
 * - Cmd+Shift+P (Ctrl+Shift+P on Windows/Linux) to open
 * - Type to filter prompts and tools
 * - Up/Down arrows to navigate
 * - Enter to select
 * - Escape to close
 */

import { useEffect, useState, useRef, useMemo } from 'react';
import { Card } from './ui/card';
import { Input } from './ui/input';
import { X, Search, FileText, Wrench } from 'lucide-react';
import type { MCPPrompt, MCPTool } from '../lib/api-client';

// Unified item type for the palette
type PaletteItem =
  | { type: 'prompt'; data: MCPPrompt }
  | { type: 'tool'; data: MCPTool };

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  prompts: MCPPrompt[];
  tools: MCPTool[];
  onSelectPrompt: (prompt: MCPPrompt) => void;
  onSelectTool: (tool: MCPTool) => void;
  isLoading?: boolean;
}

export function CommandPalette({
  isOpen,
  onClose,
  prompts,
  tools,
  onSelectPrompt,
  onSelectTool,
  isLoading = false,
}: CommandPaletteProps): JSX.Element | null {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Combine prompts and tools into a single list
  const allItems = useMemo((): PaletteItem[] => {
    const promptItems: PaletteItem[] = prompts.map((prompt) => ({
      type: 'prompt' as const,
      data: prompt,
    }));
    const toolItems: PaletteItem[] = tools.map((tool) => ({
      type: 'tool' as const,
      data: tool,
    }));
    return [...promptItems, ...toolItems];
  }, [prompts, tools]);

  // Filter items based on search query
  const filteredItems = useMemo(() => {
    return allItems.filter((item) => {
      const query = searchQuery.toLowerCase();
      const name = item.data.name.toLowerCase();
      const description = item.data.description?.toLowerCase() || '';
      return name.includes(query) || description.includes(query);
    });
  }, [allItems, searchQuery]);

  // Reset state when opened
  useEffect(() => {
    if (isOpen) {
      setSearchQuery('');
      setSelectedIndex(0);
      // Focus search input when opened
      setTimeout(() => searchInputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Reset selected index when filtered items change
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
            prev < filteredItems.length - 1 ? prev + 1 : prev
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredItems[selectedIndex]) {
            const item = filteredItems[selectedIndex];
            if (item.type === 'prompt') {
              onSelectPrompt(item.data);
            } else {
              onSelectTool(item.data);
            }
            onClose();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredItems, selectedIndex, onSelectPrompt, onSelectTool, onClose]);

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
              placeholder="Search prompts & tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Items list */}
        <div
          ref={listRef}
          className="max-h-[400px] overflow-y-auto"
        >
          {isLoading ? (
            <div className="p-8 text-center text-gray-500">
              Loading...
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              {searchQuery ? 'No items found' : 'No prompts or tools available'}
            </div>
          ) : (
            filteredItems.map((item, index) => {
              const isPrompt = item.type === 'prompt';
              const Icon = isPrompt ? FileText : Wrench;
              const iconColor = isPrompt ? 'text-blue-500' : 'text-purple-500';

              // Truncate description: remove newlines and limit to 200 chars
              const truncateDescription = (desc?: string): string | undefined => {
                if (!desc) return undefined;
                const singleLine = desc.replace(/\s+/g, ' ').trim();
                return singleLine.length > 200 ? singleLine.slice(0, 200) + '...' : singleLine;
              };

              return (
                <button
                  key={`${item.type}-${item.data.name}`}
                  onClick={() => {
                    if (item.type === 'prompt') {
                      onSelectPrompt(item.data);
                    } else {
                      onSelectTool(item.data);
                    }
                    onClose();
                  }}
                  className={`w-full text-left px-4 py-3 border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors ${
                    index === selectedIndex
                      ? 'bg-blue-50 dark:bg-blue-900/20 border-l-2 border-l-blue-500'
                      : ''
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <Icon className={`h-4 w-4 mt-0.5 flex-shrink-0 ${iconColor}`} />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm">{item.data.name}</div>
                      {truncateDescription(item.data.description) && (
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {truncateDescription(item.data.description)}
                        </div>
                      )}
                      {isPrompt && item.data.arguments && item.data.arguments.length > 0 && (
                        <div className="text-xs text-gray-400 mt-1">
                          Arguments: {item.data.arguments.map(arg => arg.name).join(', ')}
                        </div>
                      )}
                      {!isPrompt && item.data.input_schema?.required && item.data.input_schema.required.length > 0 && (
                        <div className="text-xs text-gray-400 mt-1">
                          Required: {item.data.input_schema.required.join(', ')}
                        </div>
                      )}
                    </div>
                  </div>
                </button>
              );
            })
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

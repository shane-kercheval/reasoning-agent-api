/**
 * TabBar component - horizontal tab strip for managing multiple chats.
 *
 * Browser-style tabs with close buttons, allowing users to have multiple
 * conversations open simultaneously.
 */

import { X, Plus } from 'lucide-react';
import { useTabsStore } from '../../store/tabs-store';
import { cn } from '../../lib/utils';
import { Button } from '../ui/button';

export interface TabBarProps {
  onNewTab: () => void;
}

/**
 * Horizontal tab bar for managing chat tabs.
 *
 * @example
 * ```tsx
 * <TabBar onNewTab={handleNewTab} />
 * ```
 */
export function TabBar({ onNewTab }: TabBarProps): JSX.Element {
  const tabs = useTabsStore((state) => state.tabs);
  const activeTabId = useTabsStore((state) => state.activeTabId);
  const switchTab = useTabsStore((state) => state.switchTab);
  const removeTab = useTabsStore((state) => state.removeTab);

  const handleCloseTab = (tabId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    removeTab(tabId);
  };

  return (
    <div className="flex items-center border-b bg-muted/30 overflow-x-auto">
      {/* Tabs */}
      <div className="flex flex-1 min-w-0">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTabId;

          return (
            <div
              key={tab.id}
              className={cn(
                'group relative flex items-center gap-2 px-4 py-2 border-r',
                'cursor-pointer transition-colors min-w-[120px] max-w-[240px]',
                isActive
                  ? 'bg-background border-b-2 border-b-primary'
                  : 'bg-muted/50 hover:bg-muted border-b-2 border-b-transparent',
              )}
              onClick={() => switchTab(tab.id)}
            >
              {/* Tab title */}
              <span className="flex-1 truncate text-xs font-medium">{tab.title}</span>

              {/* Close button */}
              {tabs.length > 1 && (
                <button
                  onClick={(e) => handleCloseTab(tab.id, e)}
                  className={cn(
                    'flex-shrink-0 p-0.5 rounded hover:bg-muted-foreground/20',
                    'transition-opacity',
                    isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100',
                  )}
                  aria-label="Close tab"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* New tab button */}
      <Button
        size="icon"
        variant="ghost"
        onClick={onNewTab}
        className="h-8 w-8 mx-1 flex-shrink-0"
        title="New tab"
      >
        <Plus className="h-4 w-4" />
      </Button>
    </div>
  );
}

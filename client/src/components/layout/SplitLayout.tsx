/**
 * SplitLayout component - main application layout with sidebar.
 *
 * Provides a split-pane layout with resizable sidebar for settings
 * and main content area for chat.
 */

import * as React from 'react';
import { cn } from '../../lib/utils';

export interface SplitLayoutProps {
  /** Sidebar content (settings, controls, etc.) */
  sidebar: React.ReactNode;
  /** Main content area (chat, messages, etc.) */
  children: React.ReactNode;
  /** Optional className for custom styling */
  className?: string;
}

/**
 * Split-pane layout with 1/3 sidebar, 2/3 main content.
 *
 * @example
 * ```tsx
 * <SplitLayout sidebar={<SettingsPanel />}>
 *   <ChatArea />
 * </SplitLayout>
 * ```
 */
export function SplitLayout({ sidebar, children, className }: SplitLayoutProps): JSX.Element {
  return (
    <div className={cn('flex h-screen overflow-hidden', className)}>
      {/* Sidebar - Settings Panel */}
      <aside className="w-80 border-r bg-muted/30 overflow-y-auto flex-shrink-0">
        {sidebar}
      </aside>

      {/* Main Content - Chat Area */}
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}

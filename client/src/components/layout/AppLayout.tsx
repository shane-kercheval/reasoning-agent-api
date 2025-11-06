/**
 * AppLayout component - main application layout with three panels.
 *
 * Layout structure (ChatGPT-style):
 * - Left: Collapsible conversations sidebar (resizable)
 * - Center: Chat area
 * - Right: Collapsible settings sidebar
 *
 * Features:
 * - Collapsible sidebars with toggle buttons
 * - Resizable conversation sidebar
 * - Responsive design
 */

import { useState, useRef, useEffect } from 'react';
import { PanelLeftClose, PanelLeftOpen, Settings, X } from 'lucide-react';
import { Button } from '../ui/button';

export interface AppLayoutProps {
  /** Conversation list sidebar content */
  conversationsSidebar: React.ReactNode;
  /** Settings panel content */
  settingsSidebar: React.ReactNode;
  /** Tab bar for managing multiple chats (optional) */
  tabBar?: React.ReactNode;
  /** Main chat content */
  children: React.ReactNode;
}

/**
 * Main application layout with collapsible/resizable sidebars.
 *
 * @example
 * ```tsx
 * <AppLayout
 *   conversationsSidebar={<ConversationList {...} />}
 *   settingsSidebar={<SettingsPanel {...} />}
 * >
 *   <ChatLayout {...} />
 * </AppLayout>
 * ```
 */
export function AppLayout({
  conversationsSidebar,
  settingsSidebar,
  tabBar,
  children,
}: AppLayoutProps): JSX.Element {
  const [isConversationsOpen, setIsConversationsOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [conversationsWidth, setConversationsWidth] = useState(280); // Default 280px
  const isResizing = useRef(false);

  const minWidth = 200;
  const maxWidth = 400;

  // Handle mouse move for resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;

      const newWidth = Math.max(minWidth, Math.min(maxWidth, e.clientX));
      setConversationsWidth(newWidth);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizing.current) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const handleResizeStart = () => {
    isResizing.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Conversations Sidebar (Left) */}
      {isConversationsOpen && (
        <>
          <div
            className="flex-shrink-0 overflow-hidden"
            style={{ width: `${conversationsWidth}px` }}
          >
            {conversationsSidebar}
          </div>

          {/* Resize handle */}
          <div
            className="w-1 cursor-col-resize hover:bg-primary/20 active:bg-primary/30 transition-colors flex-shrink-0"
            onMouseDown={handleResizeStart}
            title="Drag to resize"
          />
        </>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar with toggle buttons */}
        <div className="flex items-center gap-2 px-4 py-2 border-b bg-background">
          {/* Toggle conversations button */}
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setIsConversationsOpen(!isConversationsOpen)}
            title={isConversationsOpen ? 'Hide conversations' : 'Show conversations'}
            className="h-8 w-8"
          >
            {isConversationsOpen ? (
              <PanelLeftClose className="h-4 w-4" />
            ) : (
              <PanelLeftOpen className="h-4 w-4" />
            )}
          </Button>

          {/* Toggle settings button */}
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setIsSettingsOpen(!isSettingsOpen)}
            title={isSettingsOpen ? 'Hide settings' : 'Show settings'}
            className="h-8 w-8"
          >
            {isSettingsOpen ? <X className="h-4 w-4" /> : <Settings className="h-4 w-4" />}
          </Button>
        </div>

        {/* Tab bar (if provided) */}
        {tabBar}

        {/* Chat content */}
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>

      {/* Settings Sidebar (Right) */}
      {isSettingsOpen && (
        <div className="w-80 flex-shrink-0 overflow-hidden border-l">
          {settingsSidebar}
        </div>
      )}
    </div>
  );
}

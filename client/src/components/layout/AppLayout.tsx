/**
 * AppLayout component - main application layout with two panels.
 *
 * Layout structure:
 * - Left: Collapsible conversations sidebar (resizable)
 * - Center: Chat area with integrated settings
 *
 * Features:
 * - Collapsible conversations sidebar with toggle button
 * - Resizable conversation sidebar
 * - Responsive design
 */

import { useState, useRef, useEffect } from 'react';

export interface AppLayoutProps {
  /** Conversation list sidebar content */
  conversationsSidebar: React.ReactNode;
  /** Tab bar for managing multiple chats (optional) */
  tabBar?: React.ReactNode;
  /** Main chat content */
  children: React.ReactNode;
  /** Whether conversations sidebar is open (defaults to true) */
  isConversationsOpen?: boolean;
}

/**
 * Main application layout with collapsible/resizable conversations sidebar.
 *
 * Toggle button is provided by TabBar component.
 *
 * @example
 * ```tsx
 * <AppLayout
 *   conversationsSidebar={<ConversationList {...} />}
 *   isConversationsOpen={isOpen}
 * >
 *   <ChatLayout {...} />
 * </AppLayout>
 * ```
 */
export function AppLayout({
  conversationsSidebar,
  tabBar,
  children,
  isConversationsOpen = true,
}: AppLayoutProps): JSX.Element {
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
        {/* Tab bar (if provided) */}
        {tabBar}

        {/* Chat content */}
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>
    </div>
  );
}

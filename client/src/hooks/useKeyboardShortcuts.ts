import { useEffect } from 'react';

/**
 * Keyboard shortcut configuration
 */
export interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  meta?: boolean;  // Command key on Mac
  shift?: boolean;
  alt?: boolean;
  handler: (event: KeyboardEvent) => void;
  preventDefault?: boolean;
  allowInInputFields?: boolean;  // Allow this shortcut even when focused in input/textarea
}

/**
 * Platform detection utilities
 */
export const isMac = (): boolean => {
  return typeof navigator !== 'undefined' && navigator.platform.toUpperCase().indexOf('MAC') >= 0;
};

/**
 * Check if a keyboard event matches a shortcut configuration
 */
const matchesShortcut = (event: KeyboardEvent, shortcut: ShortcutConfig): boolean => {
  // Key must match (case-insensitive)
  if (event.key.toLowerCase() !== shortcut.key.toLowerCase()) {
    return false;
  }

  // Check modifiers
  const ctrlMatch = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;
  const metaMatch = shortcut.meta ? event.metaKey : !event.metaKey;
  const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
  const altMatch = shortcut.alt ? event.altKey : !event.altKey;

  return ctrlMatch && metaMatch && shiftMatch && altMatch;
};

/**
 * Custom hook for registering keyboard shortcuts
 *
 * @param shortcuts - Array of shortcut configurations
 * @param enabled - Whether shortcuts are enabled (default: true)
 *
 * @example
 * ```tsx
 * useKeyboardShortcuts([
 *   {
 *     key: 'n',
 *     meta: true, // Cmd on Mac, Ctrl on Windows/Linux
 *     handler: () => console.log('New item'),
 *     preventDefault: true
 *   }
 * ]);
 * ```
 */
export const useKeyboardShortcuts = (
  shortcuts: ShortcutConfig[],
  enabled: boolean = true
): void => {
  useEffect(() => {
    if (!enabled) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't intercept shortcuts when user is typing in an input/textarea
      // (unless the shortcut explicitly wants to)
      const target = event.target as HTMLElement;
      const isInputField = target.tagName === 'INPUT' ||
                          target.tagName === 'TEXTAREA' ||
                          target.isContentEditable;

      for (const shortcut of shortcuts) {
        if (matchesShortcut(event, shortcut)) {
          // Allow shortcuts in input fields if:
          // 1. They have modifiers (Cmd/Ctrl/Alt/Shift)
          // 2. They explicitly allow it via allowInInputFields flag
          const hasModifier = shortcut.ctrl || shortcut.meta || shortcut.shift || shortcut.alt;
          const allowedInInput = hasModifier || shortcut.allowInInputFields;

          if (isInputField && !allowedInInput) {
            continue;
          }

          if (shortcut.preventDefault) {
            event.preventDefault();
          }

          shortcut.handler(event);
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [shortcuts, enabled]);
};

/**
 * Helper to create cross-platform shortcuts (Cmd on Mac, Ctrl on Windows/Linux)
 *
 * @example
 * ```tsx
 * const shortcuts = [
 *   createCrossPlatformShortcut('n', handleNew),
 *   createCrossPlatformShortcut('w', handleClose, { shift: true })
 * ];
 * ```
 */
export const createCrossPlatformShortcut = (
  key: string,
  handler: (event: KeyboardEvent) => void,
  options: {
    shift?: boolean;
    alt?: boolean;
    preventDefault?: boolean;
  } = {}
): ShortcutConfig => {
  const mac = isMac();

  return {
    key,
    meta: mac,
    ctrl: !mac,
    shift: options.shift,
    alt: options.alt,
    handler,
    preventDefault: options.preventDefault ?? true
  };
};

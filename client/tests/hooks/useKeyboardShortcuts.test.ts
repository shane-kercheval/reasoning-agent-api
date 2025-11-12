/**
 * Unit tests for useKeyboardShortcuts hook
 *
 * Focus: High-value tests for complex logic
 * - Platform detection
 * - Key matching with modifiers
 * - Cross-platform shortcut creation
 * - Edge cases (input field handling)
 */

import { renderHook } from '@testing-library/react';
import { useKeyboardShortcuts, isMac, createCrossPlatformShortcut } from '../../src/hooks/useKeyboardShortcuts';

describe('useKeyboardShortcuts', () => {
  describe('Platform Detection', () => {
    const originalPlatform = navigator.platform;

    afterEach(() => {
      Object.defineProperty(navigator, 'platform', {
        value: originalPlatform,
        writable: true,
        configurable: true,
      });
    });

    it('detects Mac platform correctly', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'MacIntel',
        writable: true,
        configurable: true,
      });

      expect(isMac()).toBe(true);
    });

    it('detects non-Mac platforms correctly', () => {
      const platforms = ['Win32', 'Linux x86_64', 'Linux armv7l'];

      platforms.forEach((platform) => {
        Object.defineProperty(navigator, 'platform', {
          value: platform,
          writable: true,
          configurable: true,
        });

        expect(isMac()).toBe(false);
      });
    });

    it('handles case-insensitive platform detection', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'macintosh',
        writable: true,
        configurable: true,
      });

      expect(isMac()).toBe(true);
    });
  });

  describe('Cross-Platform Shortcut Creation', () => {
    const originalPlatform = navigator.platform;

    afterEach(() => {
      Object.defineProperty(navigator, 'platform', {
        value: originalPlatform,
        writable: true,
        configurable: true,
      });
    });

    it('creates Cmd shortcut on Mac', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'MacIntel',
        writable: true,
        configurable: true,
      });

      const handler = jest.fn();
      const shortcut = createCrossPlatformShortcut('n', handler);

      expect(shortcut.key).toBe('n');
      expect(shortcut.meta).toBe(true);
      expect(shortcut.ctrl).toBe(false);
      expect(shortcut.preventDefault).toBe(true);
    });

    it('creates Ctrl shortcut on Windows/Linux', () => {
      Object.defineProperty(navigator, 'platform', {
        value: 'Win32',
        writable: true,
        configurable: true,
      });

      const handler = jest.fn();
      const shortcut = createCrossPlatformShortcut('n', handler);

      expect(shortcut.key).toBe('n');
      expect(shortcut.meta).toBe(false);
      expect(shortcut.ctrl).toBe(true);
      expect(shortcut.preventDefault).toBe(true);
    });

    it('includes shift modifier when specified', () => {
      const handler = jest.fn();
      const shortcut = createCrossPlatformShortcut('f', handler, { shift: true });

      expect(shortcut.shift).toBe(true);
    });

    it('includes alt modifier when specified', () => {
      const handler = jest.fn();
      const shortcut = createCrossPlatformShortcut('f', handler, { alt: true });

      expect(shortcut.alt).toBe(true);
    });

    it('respects custom preventDefault option', () => {
      const handler = jest.fn();
      const shortcut = createCrossPlatformShortcut('n', handler, { preventDefault: false });

      expect(shortcut.preventDefault).toBe(false);
    });
  });

  describe('Keyboard Event Handling', () => {
    it('calls handler when shortcut matches', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const event = new KeyboardEvent('keydown', {
        key: 'n',
        metaKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('does not call handler when key does not match', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const event = new KeyboardEvent('keydown', {
        key: 'm',
        metaKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();
    });

    it('does not call handler when modifiers do not match', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      // Missing meta key
      const event = new KeyboardEvent('keydown', {
        key: 'n',
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();
    });

    it('matches shortcuts with multiple modifiers', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'f',
          meta: true,
          shift: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const event = new KeyboardEvent('keydown', {
        key: 'f',
        metaKey: true,
        shiftKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('does not match when extra modifiers are pressed', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      // Extra shift key pressed
      const event = new KeyboardEvent('keydown', {
        key: 'n',
        metaKey: true,
        shiftKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();
    });

    it('is case-insensitive for key matching', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const event = new KeyboardEvent('keydown', {
        key: 'N', // Uppercase
        metaKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).toHaveBeenCalledTimes(1);
    });
  });

  describe('Input Field Handling', () => {
    it('allows shortcuts with modifiers in input fields', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      const event = new KeyboardEvent('keydown', {
        key: 'n',
        metaKey: true,
        bubbles: true,
      });

      input.dispatchEvent(event);

      expect(handler).toHaveBeenCalledTimes(1);

      document.body.removeChild(input);
    });

    it('blocks shortcuts without modifiers in input fields by default', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'a',
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      const event = new KeyboardEvent('keydown', {
        key: 'a',
        bubbles: true,
      });

      input.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });

    it('allows shortcuts in input fields when allowInInputFields is true', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'Escape',
          handler,
          preventDefault: true,
          allowInInputFields: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      const event = new KeyboardEvent('keydown', {
        key: 'Escape',
        bubbles: true,
      });

      input.dispatchEvent(event);

      expect(handler).toHaveBeenCalledTimes(1);

      document.body.removeChild(input);
    });

    it('allows shortcuts in textareas with modifiers', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          ctrl: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts));

      const textarea = document.createElement('textarea');
      document.body.appendChild(textarea);
      textarea.focus();

      const event = new KeyboardEvent('keydown', {
        key: 'n',
        ctrlKey: true,
        bubbles: true,
      });

      textarea.dispatchEvent(event);

      expect(handler).toHaveBeenCalledTimes(1);

      document.body.removeChild(textarea);
    });
  });

  describe('Hook Lifecycle', () => {
    it('can be disabled via enabled parameter', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      renderHook(() => useKeyboardShortcuts(shortcuts, false));

      const event = new KeyboardEvent('keydown', {
        key: 'n',
        metaKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();
    });

    it('removes event listeners on unmount', () => {
      const handler = jest.fn();
      const shortcuts = [
        {
          key: 'n',
          meta: true,
          handler,
          preventDefault: true,
        },
      ];

      const { unmount } = renderHook(() => useKeyboardShortcuts(shortcuts));

      unmount();

      const event = new KeyboardEvent('keydown', {
        key: 'n',
        metaKey: true,
        bubbles: true,
      });

      window.dispatchEvent(event);

      expect(handler).not.toHaveBeenCalled();
    });
  });
});

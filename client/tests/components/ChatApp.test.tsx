/**
 * Tests for ChatApp error handling.
 *
 * Tests that stream errors are properly displayed and cleared.
 */

import { renderHook, act } from '@testing-library/react';
import { useTabsStore } from '../../src/store/tabs-store';

describe('ChatApp error handling', () => {
  beforeEach(() => {
    // Reset tabs store before each test
    useTabsStore.setState({
      tabs: [],
      activeTabId: null,
    });
  });

  describe('streamError state management', () => {
    it('should store streamError in tab state when error occurs', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        // Add a new tab
        result.current.addTab();
      });

      const tabId = result.current.tabs[0].id;

      act(() => {
        // Simulate stream error
        result.current.updateTab(tabId, {
          streamError: 'Network error occurred',
        });
      });

      const tab = result.current.tabs.find((t) => t.id === tabId);
      expect(tab?.streamError).toBe('Network error occurred');
    });

    it('should clear streamError when explicitly cleared', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab();
      });

      const tabId = result.current.tabs[0].id;

      act(() => {
        // Set error
        result.current.updateTab(tabId, { streamError: 'Error message' });
      });

      expect(result.current.tabs[0].streamError).toBe('Error message');

      act(() => {
        // Clear error
        result.current.updateTab(tabId, { streamError: null });
      });

      expect(result.current.tabs[0].streamError).toBeNull();
    });

    it('should maintain streamError independently across tabs', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab(); // Tab 1
        result.current.addTab(); // Tab 2
      });

      const tab1Id = result.current.tabs[0].id;
      const tab2Id = result.current.tabs[1].id;

      act(() => {
        // Set error only on tab 1
        result.current.updateTab(tab1Id, { streamError: 'Tab 1 error' });
      });

      const tab1 = result.current.tabs.find((t) => t.id === tab1Id);
      const tab2 = result.current.tabs.find((t) => t.id === tab2Id);

      expect(tab1?.streamError).toBe('Tab 1 error');
      expect(tab2?.streamError).toBeFalsy(); // undefined or null
    });
  });

  describe('error clearing on actions', () => {
    it('should have streamError initialized as null for new tabs', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab();
      });

      expect(result.current.tabs[0].streamError).toBeFalsy(); // undefined or null
    });

    it('should preserve streamError until explicitly cleared', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab();
      });

      const tabId = result.current.tabs[0].id;

      act(() => {
        result.current.updateTab(tabId, { streamError: 'Persistent error' });
      });

      // Update other properties without clearing error
      act(() => {
        result.current.updateTab(tabId, { input: 'New input' });
      });

      const tab = result.current.tabs.find((t) => t.id === tabId);
      expect(tab?.streamError).toBe('Persistent error');
      expect(tab?.input).toBe('New input');
    });
  });

  describe('edge cases', () => {
    it('should handle empty error message', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab();
      });

      const tabId = result.current.tabs[0].id;

      act(() => {
        result.current.updateTab(tabId, { streamError: '' });
      });

      // Empty string should be stored as-is
      expect(result.current.tabs[0].streamError).toBe('');
    });

    it('should handle setting null explicitly', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab();
      });

      const tabId = result.current.tabs[0].id;

      act(() => {
        result.current.updateTab(tabId, { streamError: 'Error' });
        result.current.updateTab(tabId, { streamError: null });
      });

      expect(result.current.tabs[0].streamError).toBeNull();
    });

    it('should not lose streamError when tab is switched', () => {
      const { result } = renderHook(() => useTabsStore());

      act(() => {
        result.current.addTab(); // Tab 1
        result.current.addTab(); // Tab 2
      });

      const tab1Id = result.current.tabs[0].id;
      const tab2Id = result.current.tabs[1].id;

      act(() => {
        // Set error on tab 1
        result.current.updateTab(tab1Id, { streamError: 'Tab 1 error' });
        // Switch to tab 2
        result.current.switchTab(tab2Id);
      });

      // Error should still be present on tab 1
      const tab1 = result.current.tabs.find((t) => t.id === tab1Id);
      expect(tab1?.streamError).toBe('Tab 1 error');
    });
  });
});

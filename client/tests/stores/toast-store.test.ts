/**
 * Tests for toast store.
 *
 * Tests toast notification store and auto-removal.
 */

import { useToastStore, useToast } from '../../src/store/toast-store';
import { renderHook, act } from '@testing-library/react';

describe('toast-store', () => {
  beforeEach(() => {
    // Reset store before each test
    const store = useToastStore.getState();
    store.clearAll();
  });

  afterEach(() => {
    // Clear any pending timers
    jest.clearAllTimers();
  });

  describe('addToast', () => {
    it('adds toast with auto-generated id', () => {
      const store = useToastStore.getState();

      store.addToast({ type: 'success', message: 'Test message' });

      const state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);
      expect(state.toasts[0].message).toBe('Test message');
      expect(state.toasts[0].type).toBe('success');
      expect(state.toasts[0].id).toBeDefined();
    });

    it('adds default duration of 3000ms', () => {
      const store = useToastStore.getState();

      store.addToast({ type: 'info', message: 'Test' });

      const state = useToastStore.getState();
      expect(state.toasts[0].duration).toBe(3000);
    });

    it('allows custom duration', () => {
      const store = useToastStore.getState();

      store.addToast({ type: 'warning', message: 'Test', duration: 5000 });

      const state = useToastStore.getState();
      expect(state.toasts[0].duration).toBe(5000);
    });

    it('adds multiple toasts', () => {
      const store = useToastStore.getState();

      store.addToast({ type: 'success', message: 'First' });
      store.addToast({ type: 'error', message: 'Second' });

      const state = useToastStore.getState();
      expect(state.toasts).toHaveLength(2);
      expect(state.toasts[0].message).toBe('First');
      expect(state.toasts[1].message).toBe('Second');
    });

    it('auto-removes toast after duration', () => {
      jest.useFakeTimers();
      const store = useToastStore.getState();

      store.addToast({ type: 'success', message: 'Test', duration: 1000 });

      let state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);

      // Fast-forward time
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      state = useToastStore.getState();
      expect(state.toasts).toHaveLength(0);

      jest.useRealTimers();
    });

    it('does not auto-remove if duration is 0', () => {
      jest.useFakeTimers();
      const store = useToastStore.getState();

      store.addToast({ type: 'info', message: 'Persistent', duration: 0 });

      let state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);

      act(() => {
        jest.advanceTimersByTime(10000);
      });

      state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);

      jest.useRealTimers();
    });
  });

  describe('removeToast', () => {
    it('removes specific toast by id', () => {
      const store = useToastStore.getState();
      store.addToast({ type: 'success', message: 'First' });
      store.addToast({ type: 'error', message: 'Second' });

      const toastId = useToastStore.getState().toasts[0].id;

      store.removeToast(toastId);

      const state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);
      expect(state.toasts[0].message).toBe('Second');
    });

    it('handles removing non-existent toast gracefully', () => {
      const store = useToastStore.getState();
      store.addToast({ type: 'info', message: 'Test' });

      store.removeToast('non-existent-id');

      const state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);
    });
  });

  describe('clearAll', () => {
    it('removes all toasts', () => {
      const store = useToastStore.getState();
      store.addToast({ type: 'success', message: 'First' });
      store.addToast({ type: 'error', message: 'Second' });
      store.addToast({ type: 'warning', message: 'Third' });

      store.clearAll();

      const state = useToastStore.getState();
      expect(state.toasts).toHaveLength(0);
    });
  });

  describe('useToast hook', () => {
    it('provides success method', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.success('Success message');
      });

      const state = useToastStore.getState();
      expect(state.toasts).toHaveLength(1);
      expect(state.toasts[0].type).toBe('success');
      expect(state.toasts[0].message).toBe('Success message');
    });

    it('provides error method', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.error('Error message');
      });

      const state = useToastStore.getState();
      expect(state.toasts[0].type).toBe('error');
      expect(state.toasts[0].message).toBe('Error message');
    });

    it('provides info method', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.info('Info message');
      });

      const state = useToastStore.getState();
      expect(state.toasts[0].type).toBe('info');
    });

    it('provides warning method', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.warning('Warning message');
      });

      const state = useToastStore.getState();
      expect(state.toasts[0].type).toBe('warning');
    });

    it('allows custom duration', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.success('Test', 5000);
      });

      const state = useToastStore.getState();
      expect(state.toasts[0].duration).toBe(5000);
    });
  });
});

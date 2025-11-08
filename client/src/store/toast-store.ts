/**
 * Zustand store for toast notifications.
 *
 * Simple toast notification system for user feedback.
 */

import { useMemo } from 'react';
import { create } from 'zustand';

// ============================================================================
// Types
// ============================================================================

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number; // milliseconds, default 3000
}

interface ToastStore {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearAll: () => void;
}

// ============================================================================
// Store
// ============================================================================

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],

  addToast: (toast) => {
    const id = `toast-${Date.now()}-${Math.random()}`;
    const newToast: Toast = {
      id,
      ...toast,
      duration: toast.duration ?? 3000,
    };

    set((state) => ({
      toasts: [...state.toasts, newToast],
    }));

    // Auto-remove after duration
    if (newToast.duration && newToast.duration > 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      }, newToast.duration);
    }
  },

  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),

  clearAll: () => set({ toasts: [] }),
}));

// ============================================================================
// Hooks
// ============================================================================

export function useToast() {
  const addToast = useToastStore((state) => state.addToast);

  // Memoize the returned object to prevent recreating on every render
  return useMemo(
    () => ({
      success: (message: string, duration?: number) => addToast({ type: 'success', message, duration: duration ?? 2000 }),
      error: (message: string, duration?: number) => addToast({ type: 'error', message, duration: duration ?? 3000 }),
      info: (message: string, duration?: number) => addToast({ type: 'info', message, duration: duration ?? 3000 }),
      warning: (message: string, duration?: number) => addToast({ type: 'warning', message, duration: duration ?? 3000 }),
    }),
    [addToast]
  );
}

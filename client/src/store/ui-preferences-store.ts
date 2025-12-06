/**
 * Zustand store for UI preferences.
 *
 * Persists UI state across browser sessions using localStorage.
 * Settings panel defaults to open for first-time users.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIPreferencesStore {
  isSettingsOpen: boolean;
  setSettingsOpen: (isOpen: boolean) => void;
  toggleSettings: () => void;
}

export const useUIPreferencesStore = create<UIPreferencesStore>()(
  persist(
    (set) => ({
      // Default to true (show settings) for first-time users
      isSettingsOpen: true,

      setSettingsOpen: (isOpen: boolean) => set({ isSettingsOpen: isOpen }),

      toggleSettings: () => set((state) => ({ isSettingsOpen: !state.isSettingsOpen })),
    }),
    {
      name: 'ui-preferences-storage',
    }
  )
);

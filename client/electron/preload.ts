import { contextBridge } from 'electron';

/**
 * Secure IPC bridge between main and renderer processes
 * Exposes a limited, safe API to the renderer process
 */

// Environment configuration
const config = {
  apiUrl: process.env.REASONING_API_URL || 'http://localhost:8000',
  apiToken: process.env.REASONING_API_TOKEN || '',
  nodeEnv: process.env.NODE_ENV || 'development',
};

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Configuration
  getConfig: () => config,

  // Platform info
  platform: process.platform,

  // Future: Add IPC methods here as needed
  // Example:
  // onUpdateAvailable: (callback: () => void) => ipcRenderer.on('update-available', callback),
  // openExternal: (url: string) => ipcRenderer.invoke('open-external', url),
});

// Type definitions for renderer process
export interface ElectronAPI {
  getConfig: () => {
    apiUrl: string;
    apiToken: string;
    nodeEnv: string;
  };
  platform: string;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

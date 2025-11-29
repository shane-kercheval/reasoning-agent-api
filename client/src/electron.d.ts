/**
 * Type declarations for Electron API exposed via preload script
 */

export interface ElectronAPI {
  getConfig: () => {
    apiUrl: string;
    toolsApiUrl: string;
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

export {};

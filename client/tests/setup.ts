import '@testing-library/jest-dom';

// Mock window.electronAPI for tests
global.window.electronAPI = {
  getConfig: () => ({
    apiUrl: 'http://localhost:8000',
    apiToken: '',
    nodeEnv: 'test',
  }),
  platform: 'darwin',
};

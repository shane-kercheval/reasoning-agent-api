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

// Suppress known act() warnings from Zustand store updates
// These warnings occur when Zustand updates state in async callbacks (finally blocks)
// which is expected behavior and doesn't indicate bugs
const originalError = console.error;
beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: An update to') &&
      args[0].includes('inside a test was not wrapped in act')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});

import { render, screen } from '@testing-library/react';
import App from '../src/App';
import { APIClientProvider } from '../src/contexts/APIClientContext';
import { APIClient } from '../src/lib/api-client';

// Mock API client for testing
const mockClient = {
  getBaseURL: () => 'http://localhost:8000',
  streamChatCompletion: jest.fn(),
  getModels: jest.fn(),
  health: jest.fn(),
} as unknown as APIClient;

describe('App', () => {
  it('renders without crashing', () => {
    render(
      <APIClientProvider client={mockClient}>
        <App />
      </APIClientProvider>
    );
    expect(screen.getByText(/Streaming Chat Demo/i)).toBeInTheDocument();
  });

  it('displays the Milestone 2 completion message', () => {
    render(
      <APIClientProvider client={mockClient}>
        <App />
      </APIClientProvider>
    );
    expect(screen.getByText(/Milestone 2: API Types & HTTP Client with SSE Streaming/i)).toBeInTheDocument();
  });

  it('shows the input form', () => {
    render(
      <APIClientProvider client={mockClient}>
        <App />
      </APIClientProvider>
    );
    const input = screen.getByPlaceholderText(/Type a message/i);
    expect(input).toBeInTheDocument();
  });
});

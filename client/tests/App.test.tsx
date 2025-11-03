import { render, screen } from '@testing-library/react';
import App from '../src/App';
import { APIClientProvider } from '../src/contexts/APIClientContext';
import { APIClient } from '../src/lib/api-client';

// Mock API client for testing
const mockClient = {
  getBaseURL: () => 'http://localhost:8000',
  streamChatCompletion: jest.fn(),
  getModels: jest.fn().mockResolvedValue(['gpt-4o-mini', 'gpt-4o']),
  health: jest.fn(),
} as unknown as APIClient;

describe('App', () => {
  it('renders without crashing', () => {
    render(
      <APIClientProvider client={mockClient}>
        <App />
      </APIClientProvider>
    );
    expect(screen.getByText(/Reasoning Agent/i)).toBeInTheDocument();
  });

  it('displays the empty state message', () => {
    render(
      <APIClientProvider client={mockClient}>
        <App />
      </APIClientProvider>
    );
    expect(screen.getByText(/Send a message to start a conversation/i)).toBeInTheDocument();
  });

  it('shows the input form', () => {
    render(
      <APIClientProvider client={mockClient}>
        <App />
      </APIClientProvider>
    );
    const input = screen.getByPlaceholderText(/Send a message/i);
    expect(input).toBeInTheDocument();
  });
});

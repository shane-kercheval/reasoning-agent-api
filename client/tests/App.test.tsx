import { render, screen } from '@testing-library/react';
import App from '../src/App';

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    expect(screen.getByText(/Streaming Chat Demo/i)).toBeInTheDocument();
  });

  it('displays the Milestone 2 completion message', () => {
    render(<App />);
    expect(screen.getByText(/Milestone 2: API Types & HTTP Client with SSE Streaming/i)).toBeInTheDocument();
  });

  it('shows the input form', () => {
    render(<App />);
    const input = screen.getByPlaceholderText(/Type a message/i);
    expect(input).toBeInTheDocument();
  });
});

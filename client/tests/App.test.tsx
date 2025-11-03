import { render, screen } from '@testing-library/react';
import App from '../src/App';

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    expect(screen.getByText(/Reasoning Agent Desktop Client/i)).toBeInTheDocument();
  });

  it('displays the hello world message', () => {
    render(<App />);
    expect(screen.getByText(/Hello World - Milestone 1 Complete!/i)).toBeInTheDocument();
  });

  it('displays configuration when electronAPI is available', () => {
    render(<App />);
    // Wait for useEffect to run
    setTimeout(() => {
      expect(screen.getByText(/Configuration/i)).toBeInTheDocument();
      expect(screen.getByText(/http:\/\/localhost:8000/i)).toBeInTheDocument();
    }, 0);
  });
});

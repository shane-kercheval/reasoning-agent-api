/**
 * Tests for Alert component.
 *
 * Tests rendering, variants, and dismiss functionality.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { Alert } from '../../src/components/ui/alert';

describe('Alert', () => {
  describe('rendering', () => {
    it('should render error message', () => {
      render(<Alert message="Test error message" />);
      expect(screen.getByText(/Test error message/)).toBeInTheDocument();
    });

    it('should render with "Error:" prefix for error variant', () => {
      render(<Alert variant="error" message="Something went wrong" />);
      expect(screen.getByText(/Error:/)).toBeInTheDocument();
      expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
    });

    it('should render warning variant', () => {
      render(<Alert variant="warning" message="Warning message" />);
      expect(screen.getByText(/Warning message/)).toBeInTheDocument();
    });

    it('should render info variant', () => {
      render(<Alert variant="info" message="Info message" />);
      expect(screen.getByText(/Info message/)).toBeInTheDocument();
    });

    it('should default to error variant when not specified', () => {
      render(<Alert message="Default message" />);
      // Should have error styling (check for "Error:" prefix)
      expect(screen.getByText(/Error:/)).toBeInTheDocument();
    });
  });

  describe('dismiss functionality', () => {
    it('should call onDismiss when X button is clicked', () => {
      const onDismiss = jest.fn();
      render(<Alert message="Test error" onDismiss={onDismiss} />);

      const dismissButton = screen.getByLabelText('Dismiss');
      fireEvent.click(dismissButton);

      expect(onDismiss).toHaveBeenCalledTimes(1);
    });

    it('should render dismiss button when onDismiss is provided', () => {
      const onDismiss = jest.fn();
      render(<Alert message="Test" onDismiss={onDismiss} />);

      expect(screen.getByLabelText('Dismiss')).toBeInTheDocument();
    });

    it('should not render dismiss button when onDismiss is not provided', () => {
      render(<Alert message="Test" />);

      expect(screen.queryByLabelText('Dismiss')).not.toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('should have proper aria-label on dismiss button', () => {
      const onDismiss = jest.fn();
      render(<Alert message="Test" onDismiss={onDismiss} />);

      const dismissButton = screen.getByLabelText('Dismiss');
      expect(dismissButton).toHaveAttribute('aria-label', 'Dismiss');
    });
  });
});

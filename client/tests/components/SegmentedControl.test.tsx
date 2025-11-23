/**
 * Tests for SegmentedControl component.
 *
 * Tests the reusable segmented button control used for
 * routing mode, context utilization, and other settings.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { SegmentedControl, type SegmentedControlOption } from '../../src/components/ui/segmented-control';

describe('SegmentedControl', () => {
  const mockOptions: readonly SegmentedControlOption<'low' | 'medium' | 'high'>[] = [
    { value: 'low', label: 'Low', description: 'Low setting description' },
    { value: 'medium', label: 'Medium', description: 'Medium setting description' },
    { value: 'high', label: 'High', description: 'High setting description' },
  ] as const;

  const mockOnChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('should render all options', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('Low')).toBeInTheDocument();
      expect(screen.getByText('Medium')).toBeInTheDocument();
      expect(screen.getByText('High')).toBeInTheDocument();
    });

    it('should render the label', () => {
      render(
        <SegmentedControl
          label="Context Utilization"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('Context Utilization')).toBeInTheDocument();
    });

    it('should display description of selected option', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="medium"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('Medium setting description')).toBeInTheDocument();
    });

    it('should update description when selection changes', () => {
      const { rerender } = render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('Low setting description')).toBeInTheDocument();

      // Change selection
      rerender(
        <SegmentedControl
          label="Test Setting"
          value="high"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('High setting description')).toBeInTheDocument();
      expect(screen.queryByText('Low setting description')).not.toBeInTheDocument();
    });
  });

  describe('selection behavior', () => {
    it('should call onChange with correct value when option is clicked', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      const mediumButton = screen.getByText('Medium');
      fireEvent.click(mediumButton);

      expect(mockOnChange).toHaveBeenCalledTimes(1);
      expect(mockOnChange).toHaveBeenCalledWith('medium');
    });

    it('should call onChange with correct value for all options', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      // Click each option
      fireEvent.click(screen.getByText('Low'));
      expect(mockOnChange).toHaveBeenLastCalledWith('low');

      fireEvent.click(screen.getByText('Medium'));
      expect(mockOnChange).toHaveBeenLastCalledWith('medium');

      fireEvent.click(screen.getByText('High'));
      expect(mockOnChange).toHaveBeenLastCalledWith('high');

      expect(mockOnChange).toHaveBeenCalledTimes(3);
    });

    it('should allow clicking already selected option', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="medium"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      const mediumButton = screen.getByText('Medium');
      fireEvent.click(mediumButton);

      // Should still call onChange even if already selected
      expect(mockOnChange).toHaveBeenCalledWith('medium');
    });
  });

  describe('styling and accessibility', () => {
    it('should apply custom className if provided', () => {
      const { container } = render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
          className="custom-class"
        />
      );

      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper.className).toContain('custom-class');
    });

    it('should have button type="button" to prevent form submission', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      const buttons = screen.getAllByRole('button');
      buttons.forEach((button) => {
        expect(button).toHaveAttribute('type', 'button');
      });
    });

    it('should have title attribute with description for accessibility', () => {
      render(
        <SegmentedControl
          label="Test Setting"
          value="low"
          options={mockOptions}
          onChange={mockOnChange}
        />
      );

      const lowButton = screen.getByText('Low');
      expect(lowButton).toHaveAttribute('title', 'Low setting description');

      const mediumButton = screen.getByText('Medium');
      expect(mediumButton).toHaveAttribute('title', 'Medium setting description');
    });
  });

  describe('context utilization specific usage', () => {
    it('should work with context utilization options', () => {
      const contextOptions: readonly SegmentedControlOption<'low' | 'medium' | 'full'>[] = [
        { value: 'low', label: 'Low', description: 'Use up to 33% of context window' },
        { value: 'medium', label: 'Medium', description: 'Use up to 66% of context window' },
        { value: 'full', label: 'Full', description: 'Use up to 100% of context window' },
      ] as const;

      const onChange = jest.fn();

      render(
        <SegmentedControl
          label="Context Utilization"
          value="low"
          options={contextOptions}
          onChange={onChange}
        />
      );

      expect(screen.getByText('Use up to 33% of context window')).toBeInTheDocument();

      fireEvent.click(screen.getByText('Full'));
      expect(onChange).toHaveBeenCalledWith('full');
    });
  });

  describe('routing mode specific usage', () => {
    it('should work with routing mode options', () => {
      const routingOptions: readonly SegmentedControlOption<'passthrough' | 'reasoning' | 'auto'>[] = [
        { value: 'passthrough', label: 'Chat', description: 'Standard chat mode. No tools. Fastest response.' },
        { value: 'reasoning', label: 'Reasoning', description: 'Use sequential reasoning with tools.' },
        { value: 'auto', label: 'Auto', description: 'Let AI decide' },
      ] as const;

      const onChange = jest.fn();

      render(
        <SegmentedControl
          label="Routing Mode"
          value="passthrough"
          options={routingOptions}
          onChange={onChange}
        />
      );

      expect(screen.getByText('Standard chat mode. No tools. Fastest response.')).toBeInTheDocument();

      fireEvent.click(screen.getByText('Reasoning'));
      expect(onChange).toHaveBeenCalledWith('reasoning');
    });
  });
});

import { render, screen, fireEvent } from '@testing-library/react';
import { ReasoningAccordion } from '../../src/components/chat/ReasoningAccordion';
import { ReasoningEventType, type ReasoningEvent } from '../../src/types/openai';

describe('ReasoningAccordion', () => {
  describe('Basic rendering', () => {
    it('renders empty accordion when no events', () => {
      const { container } = render(<ReasoningAccordion events={[]} />);
      expect(container.querySelector('.space-y-2')).toBeInTheDocument();
    });

    it('renders all events without iteration grouping', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Check weather' },
        },
        {
          type: ReasoningEventType.ReasoningComplete,
          step_iteration: 0,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      expect(screen.getByText('Planning')).toBeInTheDocument();
      expect(screen.getByText('Reasoning Complete')).toBeInTheDocument();
    });
  });

  describe('Iteration grouping', () => {
    it('groups events between ITERATION_START and ITERATION_COMPLETE', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Search for data' },
        },
        {
          type: ReasoningEventType.ToolExecutionStart,
          step_iteration: 1,
          metadata: { tools: ['web_search'] },
        },
        {
          type: ReasoningEventType.ToolResult,
          step_iteration: 1,
          metadata: { result: 'Data found' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // ITERATION_START should always be visible
      expect(screen.getByText('Iteration Start')).toBeInTheDocument();

      // Other events should be hidden by default (collapsed)
      expect(screen.queryByText('Planning')).not.toBeInTheDocument();
      expect(screen.queryByText('Tool Execution Start')).not.toBeInTheDocument();
      expect(screen.queryByText('Tool Result')).not.toBeInTheDocument();
      expect(screen.queryByText('Iteration Complete')).not.toBeInTheDocument();
    });

    it('shows nested events when ITERATION_START is clicked', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Search for data' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // Find and click the ITERATION_START button
      const iterationStartButton = screen.getByRole('button', { name: /Iteration Start/i });
      fireEvent.click(iterationStartButton);

      // Nested events should now be visible
      expect(screen.getByText('Planning')).toBeInTheDocument();
      expect(screen.getByText('Iteration Complete')).toBeInTheDocument();
    });

    it('hides nested events when ITERATION_START is clicked again', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Search for data' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      const iterationStartButton = screen.getByRole('button', { name: /Iteration Start/i });

      // Expand
      fireEvent.click(iterationStartButton);
      expect(screen.getByText('Planning')).toBeInTheDocument();

      // Collapse
      fireEvent.click(iterationStartButton);
      expect(screen.queryByText('Planning')).not.toBeInTheDocument();
    });
  });

  describe('Multiple iterations', () => {
    it('handles multiple independent iterations', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'First plan' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 2,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 2,
          metadata: { plan: 'Second plan' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 2,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // Both iteration starts should be visible
      const iterationButtons = screen.getAllByText('Iteration Start');
      expect(iterationButtons).toHaveLength(2);

      // Both sets of nested events should be hidden
      expect(screen.queryByText('First plan')).not.toBeInTheDocument();
      expect(screen.queryByText('Second plan')).not.toBeInTheDocument();
    });

    it('allows expanding one iteration without affecting others', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'First plan' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 2,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 2,
          metadata: { plan: 'Second plan' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 2,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      const iterationButtons = screen.getAllByRole('button', { name: /Iteration Start/i });

      // Expand first iteration only
      fireEvent.click(iterationButtons[0]);

      // First iteration events should be visible (checking for the Planning label, not metadata)
      const planningLabels = screen.getAllByText('Planning');
      expect(planningLabels).toHaveLength(1); // Only first iteration's Planning is visible

      // Second iteration Planning should still be hidden
      expect(planningLabels.length).toBe(1);
    });
  });

  describe('Edge cases', () => {
    it('handles incomplete iteration (no ITERATION_COMPLETE)', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'In progress' },
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // Should not crash, ITERATION_START should be visible
      expect(screen.getByText('Iteration Start')).toBeInTheDocument();

      // Planning should be hidden by default
      expect(screen.queryByText('Planning')).not.toBeInTheDocument();

      // Should be expandable
      const iterationButton = screen.getByRole('button', { name: /Iteration Start/i });
      fireEvent.click(iterationButton);
      expect(screen.getByText('Planning')).toBeInTheDocument();
    });

    it('handles events outside of iterations', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Test' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.ReasoningComplete,
          step_iteration: 0,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // REASONING_COMPLETE is outside iteration, should always be visible
      expect(screen.getByText('Reasoning Complete')).toBeInTheDocument();

      // Events within iteration should be hidden
      expect(screen.queryByText('Planning')).not.toBeInTheDocument();
    });

    it('handles events with errors within iterations', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Error,
          step_iteration: 1,
          metadata: {},
          error: 'Tool execution failed',
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // Error event should be hidden by default
      expect(screen.queryByText(/Tool execution failed/)).not.toBeInTheDocument();

      // Expand iteration
      const iterationButton = screen.getByRole('button', { name: /Iteration Start/i });
      fireEvent.click(iterationButton);

      // Error event should now be visible (there will be multiple "Error" texts)
      const errorElements = screen.getAllByText('Error');
      expect(errorElements.length).toBeGreaterThan(0);
    });

    it('handles empty metadata gracefully', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
      ];

      const { container } = render(<ReasoningAccordion events={events} />);

      // Should render without crashing
      expect(container).toBeInTheDocument();
      expect(screen.getByText('Iteration Start')).toBeInTheDocument();
    });
  });

  describe('Visual styling', () => {
    it('applies nested styling to events within iterations', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Test' },
        },
        {
          type: ReasoningEventType.IterationComplete,
          step_iteration: 1,
          metadata: {},
        },
      ];

      const { container } = render(<ReasoningAccordion events={events} />);

      // Expand the iteration
      const iterationButton = screen.getByRole('button', { name: /Iteration Start/i });
      fireEvent.click(iterationButton);

      // Check that nested events have margin styling
      const nestedEvents = container.querySelectorAll('.ml-4');
      expect(nestedEvents.length).toBeGreaterThan(0);
    });

    it('shows chevron icon that rotates on expand/collapse', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Test' },
        },
      ];

      const { container } = render(<ReasoningAccordion events={events} />);

      const iterationButton = screen.getByRole('button', { name: /Iteration Start/i });
      const chevron = container.querySelector('svg');

      // Should have rotate class when collapsed
      expect(chevron).not.toHaveClass('rotate-180');

      // Expand
      fireEvent.click(iterationButton);

      // Should have rotate-180 class when expanded
      const chevronAfterExpand = container.querySelector('svg.rotate-180');
      expect(chevronAfterExpand).toBeInTheDocument();
    });
  });

  describe('Interaction patterns', () => {
    it('preserves expansion state when re-rendering with same events', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Test' },
        },
      ];

      const { rerender } = render(<ReasoningAccordion events={events} />);

      // Expand iteration
      const iterationButton = screen.getByRole('button', { name: /Iteration Start/i });
      fireEvent.click(iterationButton);
      expect(screen.getByText('Planning')).toBeInTheDocument();

      // Re-render with same events
      rerender(<ReasoningAccordion events={events} />);

      // Should still be expanded
      expect(screen.getByText('Planning')).toBeInTheDocument();
    });

    it('allows individual events within iteration to be collapsed', () => {
      const events: ReasoningEvent[] = [
        {
          type: ReasoningEventType.IterationStart,
          step_iteration: 1,
          metadata: {},
        },
        {
          type: ReasoningEventType.Planning,
          step_iteration: 1,
          metadata: { plan: 'Detailed plan here' },
        },
      ];

      render(<ReasoningAccordion events={events} />);

      // Expand iteration
      const iterationButton = screen.getByRole('button', { name: /Iteration Start/i });
      fireEvent.click(iterationButton);

      // Planning should be visible but its details hidden (accordion collapsed)
      expect(screen.getByText('Planning')).toBeInTheDocument();

      // The metadata content should not be visible until Planning accordion is expanded
      expect(screen.queryByText('Detailed plan here')).not.toBeInTheDocument();
    });
  });
});

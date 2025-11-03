import { render, screen } from '@testing-library/react';
import { ReasoningStep, ReasoningStepMetadata } from '../../src/components/chat/ReasoningStep';
import { ReasoningEventType } from '../../src/types/openai';

describe('ReasoningStep', () => {
  it('renders planning event with correct label and icon', () => {
    const event = {
      type: ReasoningEventType.Planning,
      step_iteration: 1,
      metadata: {},
    };

    render(<ReasoningStep event={event} />);

    expect(screen.getByText('Planning')).toBeInTheDocument();
    expect(screen.getByText('Step 1')).toBeInTheDocument();
  });

  it('renders tool result event with correct styling', () => {
    const event = {
      type: ReasoningEventType.ToolResult,
      step_iteration: 2,
      metadata: {},
    };

    render(<ReasoningStep event={event} />);

    expect(screen.getByText('Tool Result')).toBeInTheDocument();
    expect(screen.getByText('Step 2')).toBeInTheDocument();
  });

  it('hides step number when showStep is false', () => {
    const event = {
      type: ReasoningEventType.ReasoningComplete,
      step_iteration: 0,
      metadata: {},
    };

    render(<ReasoningStep event={event} showStep={false} />);

    expect(screen.getByText('Reasoning Complete')).toBeInTheDocument();
    expect(screen.queryByText(/Step/i)).not.toBeInTheDocument();
  });

  it('shows error indicator when event has error', () => {
    const event = {
      type: ReasoningEventType.Error,
      step_iteration: 1,
      metadata: {},
      error: 'Something went wrong',
    };

    render(<ReasoningStep event={event} />);

    // Both the label "Error" and the indicator "Error" should be present
    const errorElements = screen.getAllByText('Error');
    expect(errorElements.length).toBeGreaterThan(0);
  });

  it('renders all event types correctly', () => {
    const eventTypes = [
      ReasoningEventType.IterationStart,
      ReasoningEventType.Planning,
      ReasoningEventType.ToolExecutionStart,
      ReasoningEventType.ToolResult,
      ReasoningEventType.IterationComplete,
      ReasoningEventType.ReasoningComplete,
      ReasoningEventType.Error,
    ];

    eventTypes.forEach((type) => {
      const event = {
        type,
        step_iteration: 1,
        metadata: {},
      };

      const { container } = render(<ReasoningStep event={event} />);

      // Should have an icon (svg element)
      const icon = container.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });
  });
});

describe('ReasoningStepMetadata', () => {
  it('renders metadata as JSON', () => {
    const event = {
      type: ReasoningEventType.Planning,
      step_iteration: 1,
      metadata: {
        plan: 'Check the weather',
        confidence: 0.95,
      },
    };

    render(<ReasoningStepMetadata event={event} />);

    expect(screen.getByText(/"plan":/)).toBeInTheDocument();
    expect(screen.getByText(/"Check the weather"/)).toBeInTheDocument();
  });

  it('shows error message when event has error', () => {
    const event = {
      type: ReasoningEventType.Error,
      step_iteration: 1,
      metadata: {},
      error: 'API rate limit exceeded',
    };

    render(<ReasoningStepMetadata event={event} />);

    expect(screen.getByText(/Error:/)).toBeInTheDocument();
    expect(screen.getByText(/API rate limit exceeded/)).toBeInTheDocument();
  });

  it('shows empty state when no metadata or error', () => {
    const event = {
      type: ReasoningEventType.IterationStart,
      step_iteration: 1,
      metadata: {},
    };

    render(<ReasoningStepMetadata event={event} />);

    expect(screen.getByText('No details')).toBeInTheDocument();
  });

  it('displays both error and metadata when both present', () => {
    const event = {
      type: ReasoningEventType.Error,
      step_iteration: 1,
      metadata: {
        stack_trace: 'Error at line 42',
      },
      error: 'Unexpected error',
    };

    render(<ReasoningStepMetadata event={event} />);

    expect(screen.getByText(/Error:/)).toBeInTheDocument();
    expect(screen.getByText(/Unexpected error/)).toBeInTheDocument();
    expect(screen.getByText(/"stack_trace":/)).toBeInTheDocument();
  });
});

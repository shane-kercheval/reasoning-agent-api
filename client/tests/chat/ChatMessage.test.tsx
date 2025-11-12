import { render, screen } from '@testing-library/react';
import { ChatMessage } from '../../src/components/chat/ChatMessage';
import { ReasoningEventType } from '../../src/types/openai';

describe('ChatMessage', () => {
  it('renders user message with correct styling', () => {
    render(<ChatMessage role="user" content="Hello, world!" />);

    expect(screen.getByText('Hello, world!')).toBeInTheDocument();
    expect(screen.getByText('U')).toBeInTheDocument();
  });

  it('renders assistant message with correct styling', () => {
    render(<ChatMessage role="assistant" content="Hi there!" />);

    expect(screen.getByText('Hi there!')).toBeInTheDocument();
    expect(screen.getByText('AI')).toBeInTheDocument();
  });

  it('renders reasoning events when provided', () => {
    const events = [
      {
        type: ReasoningEventType.Planning,
        step_iteration: 1,
        metadata: {},
      },
    ];

    render(<ChatMessage role="assistant" content="Processing..." reasoningEvents={events} />);

    expect(screen.getByText('Planning')).toBeInTheDocument();
  });

  it('shows streaming cursor when isStreaming is true', () => {
    const { container } = render(
      <ChatMessage role="assistant" content="Typing..." isStreaming={true} />,
    );

    // Check for animated cursor element
    const cursor = container.querySelector('.animate-pulse');
    expect(cursor).toBeInTheDocument();
  });

  it('does not show streaming cursor when isStreaming is false', () => {
    const { container } = render(
      <ChatMessage role="assistant" content="Complete message" isStreaming={false} />,
    );

    // Should not have cursor
    const cursor = container.querySelector('.animate-pulse');
    expect(cursor).not.toBeInTheDocument();
  });

  it('handles empty content gracefully', () => {
    render(<ChatMessage role="assistant" content="" />);

    // Should render without crashing
    expect(screen.getByText('AI')).toBeInTheDocument();
  });

  it('preserves whitespace and line breaks in content', () => {
    const multilineContent = 'Line 1\nLine 2\n  Indented line';
    const { container } = render(<ChatMessage role="user" content={multilineContent} />);

    // Check that content is rendered (with ReactMarkdown)
    const proseDiv = container.querySelector('.prose');
    expect(proseDiv).toBeInTheDocument();
    expect(proseDiv?.textContent).toContain('Line 1');
    expect(proseDiv?.textContent).toContain('Line 2');
  });

  describe('cost display', () => {
    it('displays cost when usage data includes total_cost', () => {
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        prompt_cost: 0.000015,
        completion_cost: 0.000030,
        total_cost: 0.000045,
      };

      render(<ChatMessage role="assistant" content="Response" usage={usage} />);

      expect(screen.getByText('$0.000045')).toBeInTheDocument();
    });

    it('displays cost with correct tooltip', () => {
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        prompt_cost: 0.000015,
        completion_cost: 0.000030,
        total_cost: 0.000045,
      };

      render(<ChatMessage role="assistant" content="Response" usage={usage} />);

      const costElement = screen.getByText('$0.000045');
      expect(costElement).toHaveAttribute(
        'title',
        'Total cost: $0.000045 (prompt: $0.000015 + completion: $0.000030)',
      );
    });

    it('does not display cost when total_cost is undefined', () => {
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
      };

      const { container } = render(<ChatMessage role="assistant" content="Response" usage={usage} />);

      expect(container.textContent).not.toContain('$');
    });

    it('does not display cost for user messages', () => {
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        total_cost: 0.000045,
      };

      const { container } = render(<ChatMessage role="user" content="Question" usage={usage} />);

      expect(container.textContent).not.toContain('$0.000045');
    });

    it('does not display cost when usage is undefined', () => {
      const { container } = render(<ChatMessage role="assistant" content="Response" />);

      expect(container.textContent).not.toContain('$');
    });

    it('handles zero cost correctly', () => {
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        prompt_cost: 0.0,
        completion_cost: 0.0,
        total_cost: 0.0,
      };

      render(<ChatMessage role="assistant" content="Response" usage={usage} />);

      expect(screen.getByText('$0.000000')).toBeInTheDocument();
    });

    it('does not show cost during streaming', () => {
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        total_cost: 0.000045,
      };

      const { container } = render(
        <ChatMessage role="assistant" content="Streaming..." usage={usage} isStreaming={true} />,
      );

      expect(container.textContent).not.toContain('$0.000045');
    });
  });
});

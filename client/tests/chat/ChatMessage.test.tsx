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

    // Check that the paragraph has whitespace-pre-wrap class for preserving formatting
    const paragraph = container.querySelector('.whitespace-pre-wrap');
    expect(paragraph).toBeInTheDocument();
    expect(paragraph?.textContent).toBe(multilineContent);
  });
});

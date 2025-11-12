import { parseSSEChunk, isSSEDoneLine, isValidSSEChunk } from '../src/lib/sse-parser';
import { isSSEDone, isChatCompletionChunk } from '../src/types/openai';

describe('SSE Parser', () => {
  describe('parseSSEChunk', () => {
    it('parses valid chat completion chunk', () => {
      const chunk = `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n`;

      const result = parseSSEChunk(chunk);

      expect(result).not.toBeNull();
      expect(isChatCompletionChunk(result!)).toBe(true);

      if (isChatCompletionChunk(result!)) {
        expect(result.id).toBe('chatcmpl-123');
        expect(result.object).toBe('chat.completion.chunk');
        expect(result.model).toBe('gpt-4o-mini');
        expect(result.choices[0]?.delta.content).toBe('Hello');
      }
    });

    it('parses [DONE] marker', () => {
      const chunk = 'data: [DONE]\n\n';

      const result = parseSSEChunk(chunk);

      expect(result).not.toBeNull();
      expect(isSSEDone(result!)).toBe(true);
    });

    it('handles chunk without trailing newlines', () => {
      const chunk = 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}';

      const result = parseSSEChunk(chunk);

      expect(result).not.toBeNull();
      expect(isChatCompletionChunk(result!)).toBe(true);
    });

    it('returns null for invalid format', () => {
      const chunk = 'invalid chunk';

      const result = parseSSEChunk(chunk);

      expect(result).toBeNull();
    });

    it('throws on malformed JSON', () => {
      const chunk = 'data: {invalid json}\n\n';

      expect(() => parseSSEChunk(chunk)).toThrow('Failed to parse SSE chunk JSON');
    });

    it('handles chunk with reasoning event', () => {
      const chunk = `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"reasoning_event":{"type":"planning","step_iteration":1,"metadata":{}}},"finish_reason":null}]}\n\n`;

      const result = parseSSEChunk(chunk);

      expect(result).not.toBeNull();
      expect(isChatCompletionChunk(result!)).toBe(true);

      if (isChatCompletionChunk(result!)) {
        const reasoningEvent = result.choices[0]?.delta.reasoning_event;
        expect(reasoningEvent).toBeDefined();
        expect(reasoningEvent?.type).toBe('planning');
        expect(reasoningEvent?.step_iteration).toBe(1);
      }
    });
  });

  describe('isSSEDoneLine', () => {
    it('recognizes [DONE] marker', () => {
      expect(isSSEDoneLine('data: [DONE]')).toBe(true);
      expect(isSSEDoneLine('  data: [DONE]  ')).toBe(true);
      expect(isSSEDoneLine('data: [DONE]\n\n')).toBe(true); // trim handles extra newlines
    });

    it('rejects non-DONE lines', () => {
      expect(isSSEDoneLine('data: {}')).toBe(false);
      expect(isSSEDoneLine('[DONE]')).toBe(false);
      expect(isSSEDoneLine('')).toBe(false);
    });
  });

  describe('isValidSSEChunk', () => {
    it('validates correct SSE format', () => {
      expect(isValidSSEChunk('data: {"key": "value"}\n\n')).toBe(true);
      expect(isValidSSEChunk('data: [DONE]')).toBe(true);
    });

    it('rejects invalid format', () => {
      expect(isValidSSEChunk('invalid')).toBe(false);
      expect(isValidSSEChunk('')).toBe(false);
      expect(isValidSSEChunk('{"key": "value"}')).toBe(false);
    });
  });
});

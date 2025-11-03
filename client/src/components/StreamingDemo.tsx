/**
 * Demo component showing streaming chat in action.
 *
 * This component demonstrates Milestone 2 functionality:
 * - API types working
 * - SSE streaming
 * - useStreamingChat hook
 */

import { useState } from 'react';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { createAPIClient } from '../lib/api-client';

const client = createAPIClient();

export function StreamingDemo(): JSX.Element {
  const [input, setInput] = useState('');
  const { content, isStreaming, error, reasoningEvents, sendMessage, cancel, newConversation } =
    useStreamingChat(client);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    await sendMessage(input, {
      model: 'gpt-4o-mini',
      routingMode: 'reasoning', // Use reasoning mode to see reasoning events
    });

    setInput('');
  };

  return (
    <div className="flex h-screen flex-col bg-gray-50 p-8">
      <div className="mx-auto w-full max-w-4xl">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Streaming Chat Demo</h1>
          <p className="text-gray-600">
            Milestone 2: API Types & HTTP Client with SSE Streaming ✅
          </p>
        </div>

        {/* Status */}
        {isStreaming && (
          <div className="mb-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <span className="text-blue-700 font-medium">⏳ Streaming response...</span>
              <button
                onClick={cancel}
                className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <span className="text-red-700 font-medium">❌ Error: {error}</span>
          </div>
        )}

        {/* Response Display */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4 min-h-[300px]">
          {/* Reasoning Events */}
          {reasoningEvents.length > 0 && (
            <div className="mb-4 border-b pb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                🧠 Reasoning Events ({reasoningEvents.length})
              </h3>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {reasoningEvents.map((event, i) => (
                  <div
                    key={i}
                    className="text-xs bg-purple-50 border border-purple-200 rounded p-2"
                  >
                    <div className="font-mono">
                      <span className="font-semibold">{event.type}</span> - Step{' '}
                      {event.step_iteration}
                    </div>
                    {Object.keys(event.metadata).length > 0 && (
                      <div className="text-gray-600 mt-1">
                        {JSON.stringify(event.metadata, null, 2)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Content */}
          <div className="prose prose-sm max-w-none">
            {content ? (
              <div className="whitespace-pre-wrap">{content}</div>
            ) : (
              <div className="text-gray-400 italic">
                Response will appear here as it streams...
              </div>
            )}
          </div>

          {/* Streaming cursor */}
          {isStreaming && <span className="inline-block w-2 h-4 bg-gray-900 ml-1 animate-pulse" />}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="mb-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message (e.g., 'What is 2+2?')"
              disabled={isStreaming}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
            >
              Send
            </button>
            <button
              type="button"
              onClick={newConversation}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
            >
              New
            </button>
          </div>
        </form>

        {/* Info */}
        <div className="text-xs text-gray-500 space-y-1">
          <p>
            💡 <strong>Tip:</strong> Using routing mode: <code>reasoning</code> to see reasoning
            events
          </p>
          <p>
            🔗 <strong>API URL:</strong> {client.getBaseURL()}
          </p>
          <p>
            ✅ <strong>Tests:</strong> 13 passed (SSE parser + App smoke tests)
          </p>
        </div>
      </div>
    </div>
  );
}

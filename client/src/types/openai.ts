/**
 * OpenAI-compatible API types for the Reasoning Agent API.
 *
 * These types mirror the backend API models in api/openai_protocol.py
 * and api/reasoning_models.py to ensure type safety across the stack.
 */

// ============================================================================
// Message Types
// ============================================================================

export enum MessageRole {
  System = 'system',
  User = 'user',
  Assistant = 'assistant',
  Tool = 'tool',
}

export interface Message {
  role: MessageRole;
  content: string | null;
  name?: string;
  tool_calls?: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
  refusal?: string;
}

// ============================================================================
// Request Types
// ============================================================================

export interface ChatCompletionRequest {
  model: string;
  messages: Message[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  stream?: boolean;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  logit_bias?: Record<string, number>;
  user?: string;
  response_format?: {
    type: 'text' | 'json_object' | 'json_schema';
    json_schema?: unknown;
  };
  tools?: Array<{
    type: string;
    function: {
      name: string;
      description?: string;
      parameters?: unknown;
    };
  }>;
  tool_choice?: string | Record<string, unknown>;
  parallel_tool_calls?: boolean;
  seed?: number;
  service_tier?: string;
  stream_options?: {
    include_usage?: boolean;
  };
  reasoning_effort?: 'minimal' | 'low' | 'medium' | 'high';
}

// ============================================================================
// Response Types
// ============================================================================

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: Record<string, unknown>;
  completion_tokens_details?: Record<string, unknown>;
  prompt_cost?: number;
  completion_cost?: number;
  total_cost?: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: Message;
    finish_reason: 'stop' | 'length' | 'function_call' | 'content_filter' | 'tool_calls' | null;
    logprobs?: unknown;
  }>;
  usage?: Usage;
  system_fingerprint?: string;
  service_tier?: string;
}

// ============================================================================
// Streaming Response Types
// ============================================================================

export enum ReasoningEventType {
  IterationStart = 'iteration_start',
  Planning = 'planning',
  ToolExecutionStart = 'tool_execution_start',
  ToolResult = 'tool_result',
  IterationComplete = 'iteration_complete',
  ReasoningComplete = 'reasoning_complete',
  ExternalReasoning = 'external_reasoning',
  Error = 'error',
}

export interface ReasoningEvent {
  type: ReasoningEventType;
  step_iteration: number;
  metadata: Record<string, unknown>;
  error?: string;
}

export interface Delta {
  role?: MessageRole;
  content?: string;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: {
      name?: string;
      arguments?: string;
    };
  }>;
  reasoning_event?: ReasoningEvent;
}

export interface StreamChoice {
  index: number;
  delta: Delta;
  finish_reason: 'stop' | 'length' | 'function_call' | 'content_filter' | 'tool_calls' | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: StreamChoice[];
  usage?: Usage;
  system_fingerprint?: string;
  service_tier?: string;
}

// ============================================================================
// SSE Types
// ============================================================================

export interface SSEDoneMarker {
  done: true;
}

export interface ConversationMetadata {
  type: 'conversation_metadata';
  conversationId: string;
}

export type SSEData = ChatCompletionChunk | SSEDoneMarker | ConversationMetadata;

// ============================================================================
// Model Types
// ============================================================================

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  supports_reasoning?: boolean;
}

export interface ModelsResponse {
  object: 'list';
  data: ModelInfo[];
}

// ============================================================================
// Error Types
// ============================================================================

export interface APIError {
  error: {
    message: string;
    type: string;
    param?: string;
    code?: string;
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

export function isSSEDone(data: SSEData): data is SSEDoneMarker {
  return 'done' in data && data.done === true;
}

export function isChatCompletionChunk(data: SSEData): data is ChatCompletionChunk {
  return 'object' in data && data.object === 'chat.completion.chunk';
}

export function isConversationMetadata(data: SSEData): data is ConversationMetadata {
  return 'type' in data && data.type === 'conversation_metadata';
}

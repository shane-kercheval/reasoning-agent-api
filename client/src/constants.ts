/**
 * Application constants.
 *
 * Best practice: Centralize magic strings and configuration values
 * to avoid typos and make changes easier.
 */

/**
 * Routing modes for API requests.
 */
export const RoutingMode = {
  /** Direct API call, no reasoning (fastest) */
  PASSTHROUGH: 'passthrough',
  /** Single-loop reasoning agent */
  REASONING: 'reasoning',
  /** Multi-agent orchestration */
  ORCHESTRATION: 'orchestration',
  /** Let LLM classifier decide */
  AUTO: 'auto',
} as const;

export type RoutingModeType = typeof RoutingMode[keyof typeof RoutingMode];

/**
 * Default API configuration.
 */
export const APIDefaults = {
  BASE_URL: 'http://localhost:8000',
  MODEL: 'gpt-4o-mini',
  TEMPERATURE: 0.2,
} as const;

/**
 * Application configuration.
 */
export const AppConfig = {
  /** Minimum Node.js version required */
  MIN_NODE_VERSION: '18.0.0',
  /** App name */
  APP_NAME: 'Assistant',
  /** App version (should match package.json) */
  APP_VERSION: '0.1.0',
} as const;

/**
 * Test identifiers for consistent testing.
 */
export const TestIds = {
  MESSAGE_INPUT: 'message-input',
  SEND_BUTTON: 'send-button',
  CANCEL_BUTTON: 'cancel-button',
  NEW_CONVERSATION_BUTTON: 'new-conversation-button',
  RESPONSE_CONTENT: 'response-content',
  REASONING_EVENTS: 'reasoning-events',
  ERROR_MESSAGE: 'error-message',
} as const;

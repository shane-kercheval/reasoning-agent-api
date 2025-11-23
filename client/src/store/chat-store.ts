/**
 * Chat settings type definition.
 *
 * Note: This file now only contains the ChatSettings type.
 * All state management has been moved to:
 * - Per-tab streaming state: tabs-store.ts
 * - Per-conversation settings: conversation-settings-store.ts
 */

import type { RoutingModeType } from '../constants';

export interface ChatSettings {
  model: string;
  routingMode: RoutingModeType;
  temperature: number;
  systemPrompt: string;
  reasoningEffort?: 'low' | 'medium' | 'high';
  contextUtilization?: 'low' | 'medium' | 'full';
}

/**
 * Tests for conversation settings store.
 *
 * Tests per-conversation settings persistence with FIFO eviction.
 */

import { renderHook, act } from '@testing-library/react';
import { useConversationSettingsStore } from '../../src/store/conversation-settings-store';
import { RoutingMode } from '../../src/constants';
import type { ChatSettings } from '../../src/store/chat-store';

describe('useConversationSettingsStore', () => {
  beforeEach(() => {
    const { result } = renderHook(() => useConversationSettingsStore());
    act(() => {
      result.current.clearAll();
    });
  });

  describe('getSettings', () => {
    it('returns default settings for null conversationId', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      const settings = result.current.getSettings(null);

      expect(settings.model).toBe('gpt-4o-mini');
      expect(settings.temperature).toBe(0.2);
      expect(settings.routingMode).toBe(RoutingMode.PASSTHROUGH);
      expect(settings.systemPrompt).toBe('');
    });

    it('returns default settings for unknown conversationId', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      const settings = result.current.getSettings('unknown-id');

      expect(settings.model).toBe('gpt-4o-mini');
      expect(settings.temperature).toBe(0.2);
    });

    it('returns saved settings for known conversationId', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      const customSettings: ChatSettings = {
        model: 'gpt-4',
        temperature: 0.5,
        routingMode: RoutingMode.REASONING,
        systemPrompt: 'You are helpful',
      };

      act(() => {
        result.current.saveSettings('conv-1', customSettings);
      });

      const settings = result.current.getSettings('conv-1');

      expect(settings).toEqual(customSettings);
    });
  });

  describe('saveSettings', () => {
    it('saves settings for a conversation', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      const customSettings: ChatSettings = {
        model: 'gpt-4',
        temperature: 0.8,
        routingMode: RoutingMode.AUTO,
        systemPrompt: 'Test prompt',
      };

      act(() => {
        result.current.saveSettings('conv-1', customSettings);
      });

      const settings = result.current.getSettings('conv-1');
      expect(settings).toEqual(customSettings);
    });

    it('updates settings for existing conversation', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      const initialSettings: ChatSettings = {
        model: 'gpt-4o-mini',
        temperature: 0.2,
        routingMode: RoutingMode.PASSTHROUGH,
        systemPrompt: '',
      };

      const updatedSettings: ChatSettings = {
        model: 'gpt-4',
        temperature: 0.9,
        routingMode: RoutingMode.REASONING,
        systemPrompt: 'Updated',
      };

      act(() => {
        result.current.saveSettings('conv-1', initialSettings);
      });

      act(() => {
        result.current.saveSettings('conv-1', updatedSettings);
      });

      const settings = result.current.getSettings('conv-1');
      expect(settings).toEqual(updatedSettings);
    });

    it('evicts oldest conversation when limit reached', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      act(() => {
        for (let i = 0; i < 1001; i++) {
          result.current.saveSettings(`conv-${i}`, {
            model: 'gpt-4o-mini',
            temperature: 0.2,
            routingMode: RoutingMode.PASSTHROUGH,
            systemPrompt: '',
          });
        }
      });

      const firstSettings = result.current.getSettings('conv-0');
      const lastSettings = result.current.getSettings('conv-1000');

      expect(firstSettings.model).toBe('gpt-4o-mini');
      expect(lastSettings.model).toBe('gpt-4o-mini');
      expect(result.current.conversationSettings['conv-0']).toBeUndefined();
      expect(result.current.conversationSettings['conv-1000']).toBeDefined();
    });
  });

  describe('clearSettings', () => {
    it('removes settings for a conversation', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      const customSettings: ChatSettings = {
        model: 'gpt-4',
        temperature: 0.5,
        routingMode: RoutingMode.REASONING,
        systemPrompt: 'Test',
      };

      act(() => {
        result.current.saveSettings('conv-1', customSettings);
      });

      expect(result.current.conversationSettings['conv-1']).toBeDefined();

      act(() => {
        result.current.clearSettings('conv-1');
      });

      expect(result.current.conversationSettings['conv-1']).toBeUndefined();

      const settings = result.current.getSettings('conv-1');
      expect(settings.model).toBe('gpt-4o-mini');
    });
  });

  describe('clearAll', () => {
    it('removes all conversation settings', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      act(() => {
        result.current.saveSettings('conv-1', {
          model: 'gpt-4',
          temperature: 0.5,
          routingMode: RoutingMode.REASONING,
          systemPrompt: '',
        });
        result.current.saveSettings('conv-2', {
          model: 'gpt-4o',
          temperature: 0.7,
          routingMode: RoutingMode.AUTO,
          systemPrompt: '',
        });
      });

      expect(Object.keys(result.current.conversationSettings)).toHaveLength(2);

      act(() => {
        result.current.clearAll();
      });

      expect(Object.keys(result.current.conversationSettings)).toHaveLength(0);
    });
  });

  describe('FIFO eviction', () => {
    it('evicts oldest conversation based on timestamp', () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      act(() => {
        result.current.saveSettings('conv-old', {
          model: 'gpt-4o-mini',
          temperature: 0.2,
          routingMode: RoutingMode.PASSTHROUGH,
          systemPrompt: 'old',
        });
      });

      act(() => {
        for (let i = 0; i < 1000; i++) {
          result.current.saveSettings(`conv-${i}`, {
            model: 'gpt-4o-mini',
            temperature: 0.2,
            routingMode: RoutingMode.PASSTHROUGH,
            systemPrompt: '',
          });
        }
      });

      expect(result.current.conversationSettings['conv-old']).toBeUndefined();
      expect(result.current.conversationSettings['conv-999']).toBeDefined();
    });

    it('updates timestamp when settings are re-saved', async () => {
      const { result } = renderHook(() => useConversationSettingsStore());

      act(() => {
        result.current.saveSettings('conv-1', {
          model: 'gpt-4o-mini',
          temperature: 0.2,
          routingMode: RoutingMode.PASSTHROUGH,
          systemPrompt: 'first',
        });
      });

      const firstTimestamp = result.current.conversationSettings['conv-1'].timestamp;

      await new Promise(resolve => setTimeout(resolve, 10));

      act(() => {
        result.current.saveSettings('conv-1', {
          model: 'gpt-4',
          temperature: 0.5,
          routingMode: RoutingMode.REASONING,
          systemPrompt: 'updated',
        });
      });

      const secondTimestamp = result.current.conversationSettings['conv-1'].timestamp;

      expect(secondTimestamp).toBeGreaterThan(firstTimestamp);
    });
  });
});

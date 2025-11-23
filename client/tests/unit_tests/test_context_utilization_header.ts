/**
 * Tests for context utilization header and settings priority.
 *
 * Critical tests that would have caught the bug where new conversations
 * were using default settings instead of tab.settings (UI selections).
 */

import { useTabsStore } from '../../src/store/tabs-store';
import { useConversationSettingsStore } from '../../src/store/conversation-settings-store';
import type { ChatSettings } from '../../src/store/chat-store';

describe('Context Utilization Settings Priority', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset stores
    useTabsStore.setState({ tabs: [], activeTabId: null });
    useConversationSettingsStore.setState({ conversationSettings: {} });
  });

  describe('tab.settings priority for new conversations', () => {
    it('should use tab.settings when conversationId is null (new conversation)', () => {
      // This test would have caught the bug where first message used defaults
      // instead of UI-selected settings

      const tabSettings: ChatSettings = {
        model: 'gpt-4o',
        temperature: 0.5,
        routingMode: 'passthrough',
        systemPrompt: '',
        reasoningEffort: 'high',
        contextUtilization: 'low', // User selected LOW in UI
      };

      // Create tab with null conversationId (new conversation) but with settings
      useTabsStore.setState({
        tabs: [
          {
            id: 'tab-1',
            conversationId: null, // New conversation
            title: 'New Chat',
            messages: [],
            input: '',
            isStreaming: false,
            settings: tabSettings, // UI selections stored here
          },
        ],
        activeTabId: 'tab-1',
      });

      const tab = useTabsStore.getState().tabs[0];

      // When sending first message, should use tab.settings NOT defaults
      const settingsToUse = tab.settings || useConversationSettingsStore.getState().getSettings(tab.conversationId);

      expect(settingsToUse.contextUtilization).toBe('low'); // NOT 'full' (default)
      expect(settingsToUse.reasoningEffort).toBe('high');
      expect(settingsToUse.temperature).toBe(0.5);
    });

    it('should use conversation settings store when conversationId exists', () => {
      // After first message, conversation gets ID and settings migrate

      const conversationId = 'conv-123';

      // Save to conversation settings store
      useConversationSettingsStore.getState().saveSettings(conversationId, {
        model: 'gpt-4o',
        temperature: 0.7,
        routingMode: 'reasoning',
        systemPrompt: '',
        reasoningEffort: 'medium',
        contextUtilization: 'medium',
      });

      // Tab now has conversationId, settings should be null (migrated)
      useTabsStore.setState({
        tabs: [
          {
            id: 'tab-1',
            conversationId, // Has ID now
            title: 'Chat with ID',
            messages: [],
            input: '',
            isStreaming: false,
            settings: null, // Migrated to conversation settings
          },
        ],
        activeTabId: 'tab-1',
      });

      const tab = useTabsStore.getState().tabs[0];
      const settingsToUse = tab.settings || useConversationSettingsStore.getState().getSettings(tab.conversationId);

      expect(settingsToUse.contextUtilization).toBe('medium');
      expect(settingsToUse.reasoningEffort).toBe('medium');
    });

    it('should prioritize tab.settings over conversation settings if both exist', () => {
      // Edge case: tab.settings takes priority

      const conversationId = 'conv-123';

      // Save to conversation settings store
      useConversationSettingsStore.getState().saveSettings(conversationId, {
        model: 'gpt-4o',
        temperature: 0.7,
        routingMode: 'reasoning',
        systemPrompt: '',
        reasoningEffort: 'low',
        contextUtilization: 'full',
      });

      // But tab also has local settings (unusual, but possible during migration)
      const tabSettings: ChatSettings = {
        model: 'gpt-4o-mini',
        temperature: 0.5,
        routingMode: 'passthrough',
        systemPrompt: '',
        reasoningEffort: 'high',
        contextUtilization: 'low',
      };

      useTabsStore.setState({
        tabs: [
          {
            id: 'tab-1',
            conversationId,
            title: 'Chat',
            messages: [],
            input: '',
            isStreaming: false,
            settings: tabSettings, // Local settings exist
          },
        ],
        activeTabId: 'tab-1',
      });

      const tab = useTabsStore.getState().tabs[0];
      const settingsToUse = tab.settings || useConversationSettingsStore.getState().getSettings(tab.conversationId);

      // Should use tab.settings (priority)
      expect(settingsToUse.contextUtilization).toBe('low'); // From tab.settings
      expect(settingsToUse.model).toBe('gpt-4o-mini'); // From tab.settings
    });
  });

  describe('settings migration when conversation gets ID', () => {
    it('should migrate tab.settings to conversation settings when ID is assigned', () => {
      // Simulates what happens after first message completes

      const tabSettings: ChatSettings = {
        model: 'gpt-4o',
        temperature: 0.5,
        routingMode: 'passthrough',
        systemPrompt: '',
        reasoningEffort: 'high',
        contextUtilization: 'low',
      };

      // Initial state: new conversation
      useTabsStore.setState({
        tabs: [
          {
            id: 'tab-1',
            conversationId: null,
            title: 'New Chat',
            messages: [],
            input: '',
            isStreaming: false,
            settings: tabSettings,
          },
        ],
        activeTabId: 'tab-1',
      });

      // After first message: conversation gets ID
      const newConversationId = 'conv-new-123';

      // Migrate settings
      useConversationSettingsStore.getState().saveSettings(newConversationId, tabSettings);

      // Update tab
      useTabsStore.setState({
        tabs: [
          {
            id: 'tab-1',
            conversationId: newConversationId, // Now has ID
            title: 'New Chat',
            messages: [],
            input: '',
            isStreaming: false,
            settings: null, // Cleared after migration
          },
        ],
        activeTabId: 'tab-1',
      });

      // Verify settings persisted correctly
      const retrievedSettings = useConversationSettingsStore.getState().getSettings(newConversationId);

      expect(retrievedSettings.contextUtilization).toBe('low');
      expect(retrievedSettings.reasoningEffort).toBe('high');
      expect(retrievedSettings.temperature).toBe(0.5);
    });
  });

  describe('default fallback behavior', () => {
    it('should use defaults when no tab.settings and no conversationId', () => {
      // Edge case: new tab without explicit settings

      useTabsStore.setState({
        tabs: [
          {
            id: 'tab-1',
            conversationId: null,
            title: 'New Chat',
            messages: [],
            input: '',
            isStreaming: false,
            settings: null, // No settings
          },
        ],
        activeTabId: 'tab-1',
      });

      const tab = useTabsStore.getState().tabs[0];
      const settingsToUse = tab.settings || useConversationSettingsStore.getState().getSettings(tab.conversationId);

      // Should use defaults
      expect(settingsToUse.contextUtilization).toBe('full'); // Default
      expect(settingsToUse.temperature).toBe(1.0); // Default
      expect(settingsToUse.routingMode).toBe('passthrough'); // Default
    });
  });

  describe('all settings fields included', () => {
    it('should include ALL settings fields when migrating from tab to conversation', () => {
      // This test ensures we don't forget to include any field (like we did with contextUtilization)

      const completeSettings: ChatSettings = {
        model: 'gpt-4o',
        temperature: 0.5,
        routingMode: 'reasoning',
        systemPrompt: 'You are helpful',
        reasoningEffort: 'high',
        contextUtilization: 'low',
      };

      const conversationId = 'conv-123';

      // Save all fields
      useConversationSettingsStore.getState().saveSettings(conversationId, completeSettings);

      // Retrieve and verify ALL fields present
      const retrieved = useConversationSettingsStore.getState().getSettings(conversationId);

      expect(retrieved).toMatchObject({
        model: 'gpt-4o',
        temperature: 0.5,
        routingMode: 'reasoning',
        systemPrompt: 'You are helpful',
        reasoningEffort: 'high',
        contextUtilization: 'low', // CRITICAL: This was missing in the bug
      });
    });
  });
});

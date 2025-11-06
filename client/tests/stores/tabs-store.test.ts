/**
 * Tests for tabs store.
 *
 * Tests tab management, switching, and state handling.
 */

import { useTabsStore } from '../../src/store/tabs-store';

describe('tabs-store', () => {
  beforeEach(() => {
    // Reset store to initial state
    const store = useTabsStore.getState();
    store.closeAllTabs();
  });

  describe('initial state', () => {
    it('starts with one empty tab', () => {
      const state = useTabsStore.getState();

      expect(state.tabs).toHaveLength(1);
      expect(state.tabs[0].conversationId).toBeNull();
      expect(state.tabs[0].title).toBe('New Chat');
      expect(state.tabs[0].messages).toEqual([]);
    });

    it('has an active tab', () => {
      const state = useTabsStore.getState();

      expect(state.activeTabId).toBe(state.tabs[0].id);
    });
  });

  describe('addTab', () => {
    it('adds a new tab and makes it active', () => {
      const store = useTabsStore.getState();

      const newTabId = store.addTab({
        conversationId: 'conv-1',
        title: 'Test Tab',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      const state = useTabsStore.getState();
      expect(state.tabs).toHaveLength(2);
      expect(state.activeTabId).toBe(newTabId);

      const newTab = state.tabs.find((t) => t.id === newTabId);
      expect(newTab?.title).toBe('Test Tab');
      expect(newTab?.conversationId).toBe('conv-1');
    });

    it('generates unique tab IDs', () => {
      const store = useTabsStore.getState();

      const id1 = store.addTab({
        conversationId: null,
        title: 'Tab 1',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      const id2 = store.addTab({
        conversationId: null,
        title: 'Tab 2',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      expect(id1).not.toBe(id2);
    });
  });

  describe('removeTab', () => {
    it('removes a tab', () => {
      const store = useTabsStore.getState();

      const tabId = store.addTab({
        conversationId: null,
        title: 'Tab to remove',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      store.removeTab(tabId);

      const state = useTabsStore.getState();
      expect(state.tabs.find((t) => t.id === tabId)).toBeUndefined();
    });

    it('creates new empty tab if last tab is closed', () => {
      const store = useTabsStore.getState();
      const initialTabId = store.tabs[0].id;

      store.removeTab(initialTabId);

      const state = useTabsStore.getState();
      expect(state.tabs).toHaveLength(1);
      expect(state.tabs[0].title).toBe('New Chat');
      expect(state.tabs[0].conversationId).toBeNull();
    });

    it('switches to adjacent tab when active tab is closed', () => {
      const store = useTabsStore.getState();

      const tab1 = store.addTab({
        conversationId: null,
        title: 'Tab 1',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      const tab2 = store.addTab({
        conversationId: null,
        title: 'Tab 2',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      // tab2 is now active, close it
      store.removeTab(tab2);

      const state = useTabsStore.getState();
      expect(state.activeTabId).toBe(tab1);
    });
  });

  describe('switchTab', () => {
    it('changes the active tab', () => {
      const store = useTabsStore.getState();

      const newTabId = store.addTab({
        conversationId: null,
        title: 'New Tab',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      const initialTabId = store.tabs[0].id;

      store.switchTab(initialTabId);

      const state = useTabsStore.getState();
      expect(state.activeTabId).toBe(initialTabId);
    });
  });

  describe('updateTab', () => {
    it('updates tab data', () => {
      const store = useTabsStore.getState();

      const tabId = store.tabs[0].id;

      store.updateTab(tabId, {
        title: 'Updated Title',
        input: 'Some input text',
      });

      const state = useTabsStore.getState();
      const tab = state.tabs.find((t) => t.id === tabId);

      expect(tab?.title).toBe('Updated Title');
      expect(tab?.input).toBe('Some input text');
    });

    it('does not affect other tabs', () => {
      const store = useTabsStore.getState();

      const tab1Id = store.tabs[0].id;
      const tab2Id = store.addTab({
        conversationId: null,
        title: 'Tab 2',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      store.updateTab(tab1Id, { title: 'Updated Tab 1' });

      const state = useTabsStore.getState();
      const tab2 = state.tabs.find((t) => t.id === tab2Id);

      expect(tab2?.title).toBe('Tab 2');
    });
  });

  describe('findTabByConversationId', () => {
    it('finds tab with matching conversation ID', () => {
      const store = useTabsStore.getState();

      store.addTab({
        conversationId: 'conv-1',
        title: 'Conversation 1',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      const tab = store.findTabByConversationId('conv-1');

      expect(tab).toBeDefined();
      expect(tab?.conversationId).toBe('conv-1');
      expect(tab?.title).toBe('Conversation 1');
    });

    it('returns undefined if no match found', () => {
      const store = useTabsStore.getState();

      const tab = store.findTabByConversationId('nonexistent');

      expect(tab).toBeUndefined();
    });
  });

  describe('getActiveTab', () => {
    it('returns the active tab', () => {
      const store = useTabsStore.getState();

      store.updateTab(store.tabs[0].id, { title: 'Active Tab' });

      const activeTab = store.getActiveTab();

      expect(activeTab).toBeDefined();
      expect(activeTab?.title).toBe('Active Tab');
    });

    it('returns undefined if no tabs exist', () => {
      const store = useTabsStore.getState();

      // Manually clear tabs (shouldn't happen in normal use)
      useTabsStore.setState({ tabs: [], activeTabId: null });

      const activeTab = store.getActiveTab();

      expect(activeTab).toBeUndefined();
    });
  });

  describe('closeAllTabs', () => {
    it('closes all tabs and creates one new empty tab', () => {
      const store = useTabsStore.getState();

      // Add multiple tabs
      store.addTab({
        conversationId: 'conv-1',
        title: 'Tab 1',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      store.addTab({
        conversationId: 'conv-2',
        title: 'Tab 2',
        messages: [],
        input: '',
        isStreaming: false,
        streamingContent: '',
        reasoningEvents: [],
      });

      store.closeAllTabs();

      const state = useTabsStore.getState();
      expect(state.tabs).toHaveLength(1);
      expect(state.tabs[0].title).toBe('New Chat');
      expect(state.tabs[0].conversationId).toBeNull();
    });
  });
});

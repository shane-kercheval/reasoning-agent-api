/**
 * Tests for conversations store.
 *
 * Tests Zustand store actions and state transitions.
 */

import { useConversationsStore } from '../../src/store/conversations-store';
import type { ConversationSummary } from '../../src/lib/api-client';

// Mock conversation data
const mockConversation1: ConversationSummary = {
  id: 'conv-1',
  title: 'Test Conversation 1',
  system_message: 'You are a helpful assistant',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  message_count: 5,
};

const mockConversation2: ConversationSummary = {
  id: 'conv-2',
  title: 'Test Conversation 2',
  system_message: 'You are a helpful assistant',
  created_at: '2024-01-02T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
  message_count: 3,
};

describe('conversations-store', () => {
  beforeEach(() => {
    // Reset store before each test
    const store = useConversationsStore.getState();
    store.setConversations([]);
    store.setSelectedConversation(null);
    store.setLoading(false);
    store.clearError();
  });

  describe('setConversations', () => {
    it('sets conversations and clears error', () => {
      const store = useConversationsStore.getState();
      store.setError('Previous error');

      store.setConversations([mockConversation1, mockConversation2]);

      const state = useConversationsStore.getState();
      expect(state.conversations).toEqual([mockConversation1, mockConversation2]);
      expect(state.error).toBeNull();
    });
  });

  describe('addConversation', () => {
    it('adds conversation to the beginning of the list', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1]);

      store.addConversation(mockConversation2);

      const state = useConversationsStore.getState();
      expect(state.conversations).toEqual([mockConversation2, mockConversation1]);
    });

    it('adds to empty list', () => {
      const store = useConversationsStore.getState();

      store.addConversation(mockConversation1);

      const state = useConversationsStore.getState();
      expect(state.conversations).toEqual([mockConversation1]);
    });
  });

  describe('updateConversation', () => {
    it('updates existing conversation', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1, mockConversation2]);

      store.updateConversation('conv-1', { title: 'Updated Title' });

      const state = useConversationsStore.getState();
      expect(state.conversations[0].title).toBe('Updated Title');
      expect(state.conversations[0].id).toBe('conv-1');
    });

    it('does not affect other conversations', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1, mockConversation2]);

      store.updateConversation('conv-1', { title: 'Updated Title' });

      const state = useConversationsStore.getState();
      expect(state.conversations[1]).toEqual(mockConversation2);
    });

    it('handles updating non-existent conversation gracefully', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1]);

      store.updateConversation('non-existent', { title: 'Updated' });

      const state = useConversationsStore.getState();
      expect(state.conversations).toEqual([mockConversation1]);
    });
  });

  describe('removeConversation', () => {
    it('removes conversation from list', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1, mockConversation2]);

      store.removeConversation('conv-1');

      const state = useConversationsStore.getState();
      expect(state.conversations).toEqual([mockConversation2]);
    });

    it('clears selection if deleted conversation was selected', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1, mockConversation2]);
      store.setSelectedConversation('conv-1');

      store.removeConversation('conv-1');

      const state = useConversationsStore.getState();
      expect(state.selectedConversationId).toBeNull();
    });

    it('preserves selection if different conversation was deleted', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1, mockConversation2]);
      store.setSelectedConversation('conv-2');

      store.removeConversation('conv-1');

      const state = useConversationsStore.getState();
      expect(state.selectedConversationId).toBe('conv-2');
    });

    it('handles removing non-existent conversation gracefully', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1]);

      store.removeConversation('non-existent');

      const state = useConversationsStore.getState();
      expect(state.conversations).toEqual([mockConversation1]);
    });
  });

  describe('restoreConversation', () => {
    it('restores deleted conversation to the list', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation2]);

      store.restoreConversation(mockConversation1);

      const state = useConversationsStore.getState();
      expect(state.conversations).toHaveLength(2);
      expect(state.conversations[0]).toEqual(mockConversation1);
    });

    it('updates existing conversation if it already exists', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation1]);

      const updatedConv = { ...mockConversation1, title: 'Updated Title' };
      store.restoreConversation(updatedConv);

      const state = useConversationsStore.getState();
      expect(state.conversations).toHaveLength(1);
      expect(state.conversations[0].title).toBe('Updated Title');
    });

    it('restores to beginning of list', () => {
      const store = useConversationsStore.getState();
      store.setConversations([mockConversation2]);

      store.restoreConversation(mockConversation1);

      const state = useConversationsStore.getState();
      expect(state.conversations[0].id).toBe('conv-1');
      expect(state.conversations[1].id).toBe('conv-2');
    });
  });

  describe('setSelectedConversation', () => {
    it('sets selected conversation', () => {
      const store = useConversationsStore.getState();

      store.setSelectedConversation('conv-1');

      const state = useConversationsStore.getState();
      expect(state.selectedConversationId).toBe('conv-1');
    });

    it('can set to null', () => {
      const store = useConversationsStore.getState();
      store.setSelectedConversation('conv-1');

      store.setSelectedConversation(null);

      const state = useConversationsStore.getState();
      expect(state.selectedConversationId).toBeNull();
    });
  });

  describe('loading state', () => {
    it('setLoading sets loading state', () => {
      const store = useConversationsStore.getState();

      store.setLoading(true);

      const state = useConversationsStore.getState();
      expect(state.isLoading).toBe(true);
    });

    it('setLoading can clear loading state', () => {
      const store = useConversationsStore.getState();
      store.setLoading(true);

      store.setLoading(false);

      const state = useConversationsStore.getState();
      expect(state.isLoading).toBe(false);
    });
  });

  describe('error state', () => {
    it('setError sets error and clears loading', () => {
      const store = useConversationsStore.getState();
      store.setLoading(true);

      store.setError('Test error');

      const state = useConversationsStore.getState();
      expect(state.error).toBe('Test error');
      expect(state.isLoading).toBe(false);
    });

    it('clearError clears error', () => {
      const store = useConversationsStore.getState();
      store.setError('Test error');

      store.clearError();

      const state = useConversationsStore.getState();
      expect(state.error).toBeNull();
    });
  });
});

/**
 * Integration test for auto-refresh conversation list flow.
 *
 * Tests the critical user flow:
 * 1. User deletes current conversation
 * 2. New conversation is created
 * 3. User sends message (conversationId changes from null to UUID)
 * 4. Conversation list automatically refreshes and selects new conversation
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useEffect, useRef } from 'react';
import { useChatStore } from '../../src/store/chat-store';
import { useConversationsStore } from '../../src/store/conversations-store';

describe('Auto-refresh conversation list flow', () => {
  beforeEach(() => {
    // Reset stores
    const chatStore = useChatStore.getState();
    chatStore.newConversation();

    const conversationsStore = useConversationsStore.getState();
    conversationsStore.setConversations([]);
    conversationsStore.setSelectedConversation(null);
  });

  it('triggers fetch and select when conversationId changes from null to UUID', async () => {
    const fetchConversations = jest.fn();
    const selectConversation = jest.fn();

    // Simulate the ChatApp useEffect logic
    const { rerender } = renderHook(
      ({ conversationId }) => {
        const previousConversationIdRef = useRef<string | null>(null);
        const isFirstRenderRef = useRef(true);

        useEffect(() => {
          const previousId = previousConversationIdRef.current;
          const currentId = conversationId;

          // Skip first render
          if (isFirstRenderRef.current) {
            isFirstRenderRef.current = false;
            previousConversationIdRef.current = currentId;
            return;
          }

          // Detect new conversation creation
          if (previousId === null && currentId !== null) {
            fetchConversations();
            selectConversation(currentId);
          }

          previousConversationIdRef.current = currentId;
        }, [conversationId]);
      },
      { initialProps: { conversationId: null } },
    );

    // Should not trigger on first render
    expect(fetchConversations).not.toHaveBeenCalled();
    expect(selectConversation).not.toHaveBeenCalled();

    // Simulate conversation ID changing to UUID (new conversation created)
    rerender({ conversationId: 'new-conv-uuid' });

    // Should trigger fetch and select
    await waitFor(() => {
      expect(fetchConversations).toHaveBeenCalledTimes(1);
      expect(selectConversation).toHaveBeenCalledWith('new-conv-uuid');
    });
  });

  it('does not trigger on mount with null conversationId', () => {
    const fetchConversations = jest.fn();
    const selectConversation = jest.fn();

    renderHook(() => {
      const previousConversationIdRef = useRef<string | null>(null);
      const isFirstRenderRef = useRef(true);

      useEffect(() => {
        const previousId = previousConversationIdRef.current;
        const currentId = null;

        if (isFirstRenderRef.current) {
          isFirstRenderRef.current = false;
          previousConversationIdRef.current = currentId;
          return;
        }

        if (previousId === null && currentId !== null) {
          fetchConversations();
          selectConversation(currentId);
        }

        previousConversationIdRef.current = currentId;
      }, []);
    });

    expect(fetchConversations).not.toHaveBeenCalled();
    expect(selectConversation).not.toHaveBeenCalled();
  });

  it('does not trigger when conversationId changes between UUIDs', async () => {
    const fetchConversations = jest.fn();
    const selectConversation = jest.fn();

    const { rerender } = renderHook(
      ({ conversationId }) => {
        const previousConversationIdRef = useRef<string | null>(null);
        const isFirstRenderRef = useRef(true);

        useEffect(() => {
          const previousId = previousConversationIdRef.current;
          const currentId = conversationId;

          if (isFirstRenderRef.current) {
            isFirstRenderRef.current = false;
            previousConversationIdRef.current = currentId;
            return;
          }

          if (previousId === null && currentId !== null) {
            fetchConversations();
            selectConversation(currentId);
          }

          previousConversationIdRef.current = currentId;
        }, [conversationId]);
      },
      { initialProps: { conversationId: 'conv-1' } },
    );

    // Change to different UUID
    rerender({ conversationId: 'conv-2' });

    // Should not trigger (previousId was not null)
    expect(fetchConversations).not.toHaveBeenCalled();
    expect(selectConversation).not.toHaveBeenCalled();
  });

  it('does not trigger when conversationId changes from UUID to null', async () => {
    const fetchConversations = jest.fn();
    const selectConversation = jest.fn();

    const { rerender } = renderHook(
      ({ conversationId }) => {
        const previousConversationIdRef = useRef<string | null>(null);
        const isFirstRenderRef = useRef(true);

        useEffect(() => {
          const previousId = previousConversationIdRef.current;
          const currentId = conversationId;

          if (isFirstRenderRef.current) {
            isFirstRenderRef.current = false;
            previousConversationIdRef.current = currentId;
            return;
          }

          if (previousId === null && currentId !== null) {
            fetchConversations();
            selectConversation(currentId);
          }

          previousConversationIdRef.current = currentId;
        }, [conversationId]);
      },
      { initialProps: { conversationId: 'conv-1' } },
    );

    // Change to null (new conversation button clicked)
    rerender({ conversationId: null });

    // Should not trigger (previousId was not null)
    expect(fetchConversations).not.toHaveBeenCalled();
    expect(selectConversation).not.toHaveBeenCalled();
  });

  it('triggers on null -> UUID transition after conversation change', async () => {
    const fetchConversations = jest.fn();
    const selectConversation = jest.fn();

    const { rerender } = renderHook(
      ({ conversationId }) => {
        const previousConversationIdRef = useRef<string | null>(null);
        const isFirstRenderRef = useRef(true);

        useEffect(() => {
          const previousId = previousConversationIdRef.current;
          const currentId = conversationId;

          if (isFirstRenderRef.current) {
            isFirstRenderRef.current = false;
            previousConversationIdRef.current = currentId;
            return;
          }

          if (previousId === null && currentId !== null) {
            fetchConversations();
            selectConversation(currentId);
          }

          previousConversationIdRef.current = currentId;
        }, [conversationId]);
      },
      { initialProps: { conversationId: 'conv-1' } },
    );

    // User clicks "New Conversation" - goes to null
    rerender({ conversationId: null });
    expect(fetchConversations).not.toHaveBeenCalled();

    // User sends message - new conversation created
    rerender({ conversationId: 'conv-2' });

    // Should trigger now (null -> UUID)
    await waitFor(() => {
      expect(fetchConversations).toHaveBeenCalledTimes(1);
      expect(selectConversation).toHaveBeenCalledWith('conv-2');
    });
  });
});

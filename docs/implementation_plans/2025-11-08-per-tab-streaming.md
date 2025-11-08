# Implementation Plan: Independent Per-Tab Streaming

## Overview

Fix critical bug where streaming responses are shared globally across all tabs instead of being isolated per-tab. Currently, starting a stream in Tab 2 while Tab 1 is streaming causes Tab 2 to display Tab 1's content.

---

## Problem Statement

**Current Architecture (Broken)**:
- Single global `useStreamingChat` hook in `ChatApp.tsx`
- Uses global Zustand store (`useChatStore`) for all streaming state
- Only ONE stream can exist at a time across entire application
- When Tab 2 starts streaming, it either:
  - Blocks because Tab 1 is already streaming
  - Overwrites Tab 1's stream with Tab 2's content
  - Shows Tab 1's content in Tab 2 (current observed bug)

**Root Cause**:
```typescript
// ChatApp.tsx line 39 - SINGLE global hook for all tabs
const { content, isStreaming, error, reasoningEvents, usage, sendMessage, ... } =
  useStreamingChat(client);

// Line 132-148 - Uses GLOBAL content/reasoningEvents for ALL tabs
const messages = useMemo(() => {
  if ((content || isStreaming) && activeTab.isStreaming) {
    msgs.push({
      role: 'assistant',
      content: content || '',  // ← GLOBAL content!
      reasoningEvents: reasoningEvents,  // ← GLOBAL reasoningEvents!
    });
  }
}, [activeTab, content, isStreaming, reasoningEvents]);
```

**Why This Matters**:
- Multi-tab support is a core feature
- Users expect independent conversations in each tab
- Future side-by-side tab view requires concurrent streaming
- Current bug makes multi-tab feature unusable

---

## Desired Behavior

**Independent Per-Tab Streaming**:
- Each tab maintains its own streaming state completely isolated from other tabs
- Tab 1 can stream while Tab 2 streams simultaneously (concurrent streams)
- Switching between tabs shows correct content for each tab
- Canceling stream in Tab 1 doesn't affect Tab 2
- Each tab has its own AbortController for cancellation

**Future-Proof Architecture**:
- Supports side-by-side tab view with 2-4 tabs streaming concurrently
- No global streaming state (except settings panel visibility)
- Tab state is self-contained and portable

**Shared State (Intentional)**:
- Settings panel open/closed (global UI state)
- Model/routing settings (per-conversation, stored in conversation settings store)
- Everything else should be per-tab

---

## Key Design Decisions

### 1. Remove Global Streaming State
**Current**: `useChatStore` has global `streaming` object with `content`, `reasoningEvents`, `isStreaming`, etc.

**Proposed**: Remove streaming state from `useChatStore` entirely. Each tab's state lives in `useTabsStore`:
```typescript
// tabs-store.ts - Tab interface already has these fields (unused currently)
export interface Tab {
  // ... existing fields ...
  isStreaming: boolean;           // ✅ Already exists
  streamingContent: string;       // ✅ Already exists
  reasoningEvents: ReasoningEvent[]; // ✅ Already exists
  usage: Usage | null;            // ⚠️ Need to add
  streamError: string | null;     // ⚠️ Need to add
}
```

### 2. Streaming Management Strategy
**Options Considered**:

**Option A: Per-Tab Streaming Hook** (Rejected)
- Create multiple `useStreamingChat` instances (one per tab)
- Problem: React hooks can't be created dynamically per tab
- Problem: How to manage cleanup when tabs close?

**Option B: Centralized Stream Manager** (Rejected)
- Create `StreamManager` class that maps `tabId → StreamState`
- Problem: Adds complexity without clear benefit
- Problem: Still need to wire into React lifecycle

**Option C: Tab-Aware Streaming Actions** (Selected ✅)
- Create streaming action functions that take `tabId` as parameter
- Store AbortControllers in a Map keyed by `tabId`
- Update tab state via `updateTab(tabId, { streamingContent, ... })`
- Simple, explicit, easy to reason about

### 3. Stream Lifecycle Management
```typescript
// Pseudo-code showing lifecycle
async function sendMessageForTab(tabId: string, message: string) {
  // 1. Create AbortController for this tab
  const abortController = new AbortController();
  streamControllers.set(tabId, abortController);

  // 2. Update tab state: start streaming
  updateTab(tabId, {
    isStreaming: true,
    streamingContent: '',
    reasoningEvents: [],
    streamError: null,
  });

  // 3. Stream from API
  for await (const chunk of apiClient.streamChatCompletion(..., { signal: abortController.signal })) {
    // 4. Accumulate content for THIS tab only
    updateTab(tabId, {
      streamingContent: currentContent + chunk.delta.content,
    });
  }

  // 5. Complete: move streaming content to messages
  const updatedMessages = [...tab.messages, {
    role: 'assistant',
    content: tab.streamingContent,
  }];
  updateTab(tabId, {
    messages: updatedMessages,
    isStreaming: false,
    streamingContent: '',
  });

  // 6. Cleanup
  streamControllers.delete(tabId);
}
```

### 4. Concurrent Stream Support
- Map of `tabId → AbortController` allows multiple streams
- Each stream updates only its tab's state
- No locking or coordination needed (React handles updates)
- Cleanup happens independently per tab

---

## Implementation Philosophy

**Guiding Principles**:
- ✅ **Don't over-engineer**: Fix real problems, not hypothetical ones. Add complexity only when proven necessary.
- ✅ **Good architecture over premature optimization**: Focus on clean design patterns. Measure performance before optimizing.
- ✅ **Simple solutions first**: Choose the simplest approach that solves the problem correctly.
- ✅ **Ask "What's the best design pattern?"**: When in doubt, prioritize architectural quality over quick fixes.

---

## Implementation Milestones

### Milestone 1: Refactor Streaming to Tab-Aware Actions

**Goal**: Move streaming logic from global hook to tab-aware action functions

**Key Changes**:
1. **Remove global streaming state from `chat-store.ts`**:
   - Remove `streaming` object from store
   - Remove actions: `startStreaming`, `appendContent`, `addReasoningEvent`, `stopStreaming`, `clearStreaming`
   - Keep: `settings`, `conversationId` (these are still global/cross-tab)

2. **Add missing fields to Tab interface in `tabs-store.ts`**:
   ```typescript
   export interface Tab {
     // ... existing fields ...
     usage: Usage | null;        // Add this
     streamError: string | null; // Add this
   }

   interface TabsStore {
     // ... existing fields ...
     streamCleanup: ((tabId: string) => void) | null; // Add - for hook-store coordination
   }
   ```

3. **Create `client/src/hooks/useTabStreaming.ts`**:
   - New hook that returns tab-aware streaming actions
   - Takes `apiClient` as parameter
   - **Define TypeScript interface for return type**:
     ```typescript
     interface TabStreamingActions {
       sendMessageForTab: (tabId: string, message: string, options?: SendMessageOptions) => Promise<string | null>;
       regenerateForTab: (tabId: string, options?: SendMessageOptions) => Promise<string | null>;
       cancelStreamForTab: (tabId: string) => void;
     }
     ```
   - Manages Map of `tabId → AbortController`
   - Updates tab state via `updateTab(tabId, { ... })`
   - **Register cleanup function with tabs store** (for coordination with `removeTab` action)
   - **Implement React cleanup pattern**:
     ```typescript
     useEffect(() => {
       // Register cleanup function with store so removeTab can call it
       useTabsStore.setState({ streamCleanup: cleanupTabStream });

       return () => {
         // Cleanup all active streams when hook unmounts
         streamControllers.forEach(controller => controller.abort());
         streamControllers.clear();
       };
     }, []);
     ```

4. **Update `ChatApp.tsx`**:
   - Remove `useStreamingChat` hook
   - Add `useTabStreaming` hook
   - Update `handleSendMessage` to call `sendMessageForTab(activeTab.id, message, options)`
   - Update `handleCancel` to call `cancelStreamForTab(activeTab.id)`
   - Update `handleRegenerateMessage` to call `regenerateForTab(activeTab.id, options)`
   - Remove `useEffect` that watches global streaming state (lines 233-278)
   - Build messages from `activeTab.streamingContent` instead of global `content`

5. **Build Messages Logic**:
   ```typescript
   // ChatApp.tsx - messages useMemo
   const messages = useMemo(() => {
     if (!activeTab) return [];

     const msgs = [...activeTab.messages];

     // Add current streaming message if streaming
     if (activeTab.isStreaming) {
       msgs.push({
         role: 'assistant',
         content: activeTab.streamingContent || '',
         reasoningEvents: activeTab.reasoningEvents,
         usage: activeTab.usage,
       });
     }

     return msgs;
   }, [activeTab]); // Only depends on activeTab, not global state
   ```

**Success Criteria**:
- ✅ No global streaming state in `chat-store.ts`
- ✅ Each tab's streaming state isolated in `tabs-store.ts`
- ✅ `useTabStreaming` hook provides tab-aware actions
- ✅ Single tab streaming works (smoke test)
- ✅ All existing tests pass (may need updates to use new API)

**Testing Strategy**:
1. **Unit tests for `useTabStreaming` hook**:
   - Test `sendMessageForTab` updates correct tab
   - Test `cancelStreamForTab` aborts correct stream
   - Test cleanup when AbortController aborted
   - Test error handling (API failure, network error)

2. **Integration tests**:
   - Single tab streaming (basic smoke test)
   - Streaming completion moves content to messages
   - Cancel during stream works
   - Error during stream sets `streamError`

3. **Manual testing checklist**:
   - [ ] Send message in single tab, verify response
   - [ ] Cancel stream mid-response
   - [ ] Switch tabs while streaming (verify no interference)

**Risk Factors**:
- Breaking changes to hook API (tests will need updates)
- React lifecycle edge cases with AbortController cleanup
- Race conditions if tab closes mid-stream

**Dependencies**: None (foundational refactor)

---

### Milestone 2: Concurrent Multi-Tab Streaming Support

**Goal**: Enable multiple tabs to stream simultaneously without interference

**Key Changes**:
1. **Update `useTabStreaming` to support concurrent streams**:
   - Verify Map-based AbortController management works for concurrent streams
   - Ensure no shared state between streams
   - Add debug logging for concurrent stream tracking

2. **Add cleanup on tab close**:
   ```typescript
   // tabs-store.ts - removeTab action
   removeTab: (tabId) => {
     // Cancel stream if tab is streaming (uses cleanup function from M1)
     const tab = state.tabs.find(t => t.id === tabId);
     if (tab?.isStreaming && state.streamCleanup) {
       state.streamCleanup(tabId); // Call hook's cleanup function
     }

     // ... existing removeTab logic ...
   }
   ```

   **Pattern Explanation**: The hook registers its cleanup function with the store in M1. The store calls it when removing tabs. This keeps the AbortController logic in the hook while allowing the store to trigger cleanup.

3. **Verify concurrent streaming works correctly**:
   - Multiple tabs should stream independently
   - No shared state between streams
   - Each stream's AbortController isolated by `tabId`

**Success Criteria**:
- ✅ Two tabs can stream simultaneously without interference
- ✅ Tab 1 streaming doesn't affect Tab 2's content
- ✅ Closing a streaming tab cancels its stream
- ✅ Switching tabs shows correct streaming content per tab
- ✅ All concurrent streams complete successfully

**Testing Strategy**:
1. **Integration tests for concurrent streaming**:
   ```typescript
   test('two tabs can stream simultaneously', async () => {
     // Start stream in tab 1
     sendMessageForTab(tab1.id, "Message 1");

     // Start stream in tab 2 (before tab 1 completes)
     sendMessageForTab(tab2.id, "Message 2");

     // Wait for both to complete
     await waitFor(() => {
       expect(getTab(tab1.id).messages).toHaveLength(2); // user + assistant
       expect(getTab(tab2.id).messages).toHaveLength(2);
     });

     // Verify content is different and correct
     expect(getTab(tab1.id).messages[1].content).toContain("response to Message 1");
     expect(getTab(tab2.id).messages[1].content).toContain("response to Message 2");
   });
   ```

2. **Test tab closure during streaming**:
   ```typescript
   test('closing streaming tab cancels stream', async () => {
     sendMessageForTab(tab1.id, "Message");

     // Close tab mid-stream
     removeTab(tab1.id);

     // Verify AbortController was called
     expect(mockAbortController.abort).toHaveBeenCalled();
   });
   ```

3. **Manual testing checklist**:
   - [ ] Open 3 tabs, send messages in all 3 simultaneously
   - [ ] Verify each tab shows correct response
   - [ ] Switch between tabs mid-stream, verify no interference
   - [ ] Close a streaming tab, verify stream cancels
   - [ ] Start streaming in Tab 1, switch to Tab 2, send message there

**Risk Factors**:
- Race conditions with rapid tab switching
- Memory leaks if AbortControllers not cleaned up
- React re-render performance with multiple concurrent streams

**Dependencies**: Milestone 1 (foundational refactor)

---

### Milestone 3: Polish & Edge Cases

**Goal**: Handle edge cases and improve UX

**Key Changes**:
1. **Basic defensive checks**:
   - Add safety checks in streaming functions:
     ```typescript
     async function sendMessageForTab(tabId: string, message: string) {
       const tab = useTabsStore.getState().tabs.find(t => t.id === tabId);
       if (!tab) return; // Tab was deleted, abort silently

       // ... proceed with streaming ...
     }
     ```
   - Prevents errors if tab is deleted mid-operation
   - Simple guards, not complex atomic patterns (keep it simple)

2. **Error handling improvements**:
   - Show per-tab error messages (use `tab.streamError`)
   - Handle network disconnection gracefully
   - Clear error state on next successful message
   - Note: Retry mechanism not included - user can manually resend if needed

3. **UX improvements**:
   - Visual indicator when tab is streaming (tab title/icon)
   - Prevent closing tab with active stream (show confirmation)
   - Auto-scroll works correctly per tab

4. **Performance optimization** (only if needed after testing):
   - Throttle tab state updates during streaming (avoid excessive re-renders)
   - Note: Zustand batches updates automatically - only optimize if profiling shows issues
   - Measure first, optimize second

**Success Criteria**:
- ✅ All edge cases handled gracefully
- ✅ Error messages shown per-tab
- ✅ No memory leaks or performance issues
- ✅ UX smooth and responsive

**Testing Strategy**:
1. **Edge case tests**:
   - Stream in Tab 1, switch to Tab 2, close Tab 1
   - Start 10 tabs streaming simultaneously
   - Rapid tab switching during streams
   - Network error mid-stream
   - API error (500) during stream

2. **Performance tests**:
   - Measure re-render count during streaming
   - Verify no memory leaks with 20+ tabs opened/closed
   - Test with very long responses (10KB+ content)

3. **Manual testing checklist**:
   - [ ] Error message shows in correct tab
   - [ ] Closing streaming tab shows confirmation
   - [ ] Tab title shows streaming indicator
   - [ ] Auto-scroll works in each tab independently

**Risk Factors**:
- Performance degradation with many concurrent streams
- Complex state management for edge cases

**Dependencies**: Milestone 2 (concurrent streaming)

---

## Migration Notes

**Breaking Changes**:
- ✅ Breaking changes are acceptable and encouraged
- `useStreamingChat` hook replaced with `useTabStreaming`
- Global streaming state removed from `chat-store.ts`

**Backward Compatibility**:
- NOT REQUIRED (per implementation guide)
- Focus on clean, maintainable solution

**Testing Updates Required**:
- Tests using `useStreamingChat` need updates
- Mock patterns may need adjustments
- Integration tests need tab context

---

## References

**Relevant Files**:
- `client/src/hooks/useStreamingChat.ts` - Current global streaming implementation
- `client/src/store/chat-store.ts` - Global streaming state (to be removed)
- `client/src/store/tabs-store.ts` - Tab state (streaming fields already exist)
- `client/src/components/ChatApp.tsx` - Main app component using streaming

**Related Documentation**:
- React hooks lifecycle: https://react.dev/reference/react/hooks
- AbortController API: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
- Zustand store patterns: https://github.com/pmndrs/zustand

---

## Success Metrics

**Functional Requirements**:
- ✅ Multiple tabs can stream concurrently without interference
- ✅ Each tab shows correct content (no cross-contamination)
- ✅ Closing streaming tab cancels stream cleanly
- ✅ Switching tabs preserves streaming state

**Technical Requirements**:
- ✅ No global streaming state (except intentional shared state)
- ✅ No memory leaks from AbortControllers
- ✅ Clean separation of concerns (streaming logic isolated)

**Quality Requirements**:
- ✅ All tests pass (existing + new)
- ✅ Code review approval
- ✅ Manual testing checklist completed
- ✅ Documentation updated (this plan + inline comments)

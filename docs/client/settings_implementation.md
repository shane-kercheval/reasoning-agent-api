# Per-Conversation Settings Implementation

## Overview
Each conversation retains its own settings (model, temperature, routing mode, system prompt). Settings are preserved when switching between conversations and persist across app restarts.

## Architecture

### Storage Layers

**1. Tab Settings (tabs-store.ts)**
- Each tab has a `settings: ChatSettings | null` field
- Used for NEW conversations that don't have a conversationId yet
- Allows changing settings before sending first message

**2. Conversation Settings (conversation-settings-store.ts)**
- Stores settings keyed by conversationId
- FIFO eviction at 1000 conversations
- Persists to localStorage
- Used for EXISTING conversations

### Data Flow

#### New Conversation
1. Create tab with `settings: null`
2. User changes temperature → saved to `tab.settings`
3. User switches tabs → `tab.settings` preserved
4. User sends first message → conversationId created
5. Migration effect copies `tab.settings` to conversation settings
6. Tab settings cleared (`settings: null`)

#### Existing Conversation
1. Load conversation → tab created with `settings: null`
2. Restore effect loads settings from conversation settings store
3. User changes temperature → saved directly to conversation settings
4. User switches tabs → settings persisted in store
5. User switches back → settings restored from store

### Effects in ChatApp.tsx

**Restore Effect (runs when tab changes)**
```typescript
useEffect(() => {
  if (!activeTab) return;

  // Priority: tab.settings > conversation settings > defaults
  if (activeTab.settings) {
    updateSettings(activeTab.settings);
  } else if (activeTab.conversationId) {
    const conversationSettings = getSettings(activeTab.conversationId);
    updateSettings(conversationSettings);
  } else {
    const defaultSettings = getSettings(null);
    updateSettings(defaultSettings);
  }
}, [activeTab?.id]);
```

**Save Effect (runs when settings change)**
```typescript
useEffect(() => {
  if (!activeTab) return;

  if (activeTab.conversationId) {
    // Existing conversation: save to store
    saveSettings(activeTab.conversationId, settings);
  } else {
    // New conversation: save to tab
    updateTab(activeTab.id, { settings });
  }
}, [settings]);
```

**Migration Effect (runs when conversationId is assigned)**
```typescript
useEffect(() => {
  if (activeTab?.conversationId && activeTab.settings) {
    // Copy tab settings to conversation settings
    saveSettings(activeTab.conversationId, activeTab.settings);
    // Clear tab settings
    updateTab(activeTab.id, { settings: null });
  }
}, [activeTab?.conversationId, activeTab?.settings]);
```

## Example Flows

### Flow 1: New Conversation with Settings Changes
1. **New Chat** → settings = defaults (temp=0.2, gpt-4o-mini)
2. **Change temp to 0.8** → saved to tab.settings
3. **Switch to Tab B** → Tab A's tab.settings = {temp: 0.8, ...}
4. **Switch back to Tab A** → restored from tab.settings
5. **Send message** → conversationId created, settings migrated to store

### Flow 2: Existing Conversations
1. **Tab A (Conversation 1)** → temp=0.8, model=gpt-4
2. **Send message** → saved to conversation settings store
3. **Tab B (New Chat)** → temp=0.2, gpt-4o-mini (defaults)
4. **Switch to Tab A** → restored from store: temp=0.8, gpt-4
5. **Change temp to 0.5** → immediately saved to store
6. **Switch to Tab B** → still temp=0.2, gpt-4o-mini
7. **Switch to Tab A** → restored: temp=0.5, gpt-4

## Benefits

1. **Isolated Settings**: Each conversation is independent
2. **Pre-creation Settings**: Can configure before first message
3. **Persistent**: Survives app restarts (localStorage)
4. **Scalable**: FIFO eviction prevents unbounded growth
5. **Intuitive**: Settings follow the conversation

## Testing

- 205 tests passing
- 10 tests specifically for conversation-settings-store
- Covers FIFO eviction, defaults, save/restore, edge cases

/**
 * ChatApp - Main application component.
 *
 * Orchestrates the entire chat interface with multi-tab support, conversation management,
 * streaming responses, and keyboard shortcuts.
 *
 * Features:
 * - Multi-tab support for managing multiple conversations simultaneously
 * - Browser-style tabs with close buttons
 * - Each tab maintains its own conversation state
 * - Conversation list with search and filtering
 * - Settings panel for model and routing configuration
 * - Command palette for MCP prompts (Cmd+Shift+P)
 * - Keyboard shortcuts for navigation and focus management
 *   - Cmd+N: New conversation
 *   - Cmd+Shift+P: Open command palette
 *   - Cmd+W: Close current tab
 *   - Cmd+,: Toggle settings panel
 *   - Cmd+/: Show keyboard shortcuts
 *   - Cmd+Shift+F: Focus search
 *   - Cmd+Shift+[: Previous tab
 *   - Cmd+Shift+]: Next tab
 *   - Escape: Focus chat input
 */

import { useMemo, useEffect, useRef, useCallback, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTabStreaming } from '../hooks/useTabStreaming';
import { useAPIClient } from '../contexts/APIClientContext';
import { useModels } from '../hooks/useModels';
import { useMCPPrompts } from '../hooks/useMCPPrompts';
import { useMCPTools } from '../hooks/useMCPTools';
import { useConversations } from '../hooks/useConversations';
import { useLoadConversation } from '../hooks/useLoadConversation';
import { useKeyboardShortcuts, createCrossPlatformShortcut } from '../hooks/useKeyboardShortcuts';
import { ChatLayout, type ChatLayoutRef } from './ChatLayout';
import { AppLayout } from './layout/AppLayout';
import { SettingsPanel } from './settings/SettingsPanel';
import { ConversationList, type ConversationListRef } from './conversations/ConversationList';
import { TabBar } from './tabs/TabBar';
import { KeyboardShortcutsOverlay } from './KeyboardShortcutsOverlay';
import { CommandPalette } from './CommandPalette';
import { ArgumentsDialog } from './ArgumentsDialog';
import { ToolExecutionDialog } from './ToolExecutionDialog';
import { useTabsStore } from '../store/tabs-store';
import { useConversationsStore, useViewFilter, useSearchQuery } from '../store/conversations-store';
import { useConversationSettingsStore } from '../store/conversation-settings-store';
import { useToast } from '../store/toast-store';
import { processSearchResults } from '../lib/search-utils';
import type { MessageSearchResult, MCPPrompt, MCPTool, MCPToolResult, MCPPromptArgument } from '../lib/api-client';
import type { ReasoningEvent, Usage } from '../types/openai';
import type { ChatSettings } from '../store/chat-store';

export function ChatApp(): JSX.Element {
  const { client } = useAPIClient();
  const { sendMessageForTab, regenerateForTab, cancelStreamForTab } = useTabStreaming(client);
  const toast = useToast();

  // Refs for keyboard shortcuts
  const conversationListRef = useRef<ConversationListRef>(null);
  const chatLayoutRef = useRef<ChatLayoutRef>(null);

  // Search results state
  const [searchResults, setSearchResults] = useState<MessageSearchResult[] | null>(null);

  // Sidebar states (global across all tabs)
  const [isConversationsOpen, setIsConversationsOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isShortcutsOverlayOpen, setIsShortcutsOverlayOpen] = useState(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isArgumentsDialogOpen, setIsArgumentsDialogOpen] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState<MCPPrompt | null>(null);
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [selectedItemType, setSelectedItemType] = useState<'prompt' | 'tool'>('prompt');
  const [isExecuting, setIsExecuting] = useState(false);
  const [toolExecutionResult, setToolExecutionResult] = useState<MCPToolResult | null>(null);
  const [isToolResultDialogOpen, setIsToolResultDialogOpen] = useState(false);

  // Fetch available models
  const { models, isLoading: isLoadingModels } = useModels(client);

  // Fetch available MCP prompts
  const { prompts, isLoading: isLoadingPrompts } = useMCPPrompts(client);

  // Fetch available MCP tools
  const { tools, isLoading: isLoadingTools } = useMCPTools(client);

  // Tabs state - split subscriptions to minimize re-renders
  const activeTabId = useTabsStore((state) => state.activeTabId);

  // Subscribe to input separately (only MessageInput needs it)
  const input = useTabsStore((state) => {
    const tab = state.tabs.find((t) => t.id === state.activeTabId);
    return tab?.input || '';
  });

  // Subscribe to tab data that affects message display (excluding input)
  const activeTab = useTabsStore(
    useShallow((state) => {
      const tab = state.tabs.find((t) => t.id === state.activeTabId);
      if (!tab) return undefined;

      return {
        id: tab.id,
        conversationId: tab.conversationId,
        title: tab.title,
        messages: tab.messages,
        isStreaming: tab.isStreaming,
        streamingContent: tab.streamingContent,
        reasoningEvents: tab.reasoningEvents,
        usage: tab.usage,
        streamError: tab.streamError,
        settings: tab.settings,
        reasoningViewMode: tab.reasoningViewMode,
      };
    })
  );

  // Subscribe to tabs array (needed for TabBar and keyboard shortcuts)
  const tabs = useTabsStore((state) => state.tabs);

  // Store actions (stable references)
  const updateTab = useTabsStore((state) => state.updateTab);
  const addTab = useTabsStore((state) => state.addTab);
  const findTabByConversationId = useTabsStore((state) => state.findTabByConversationId);
  const switchTab = useTabsStore((state) => state.switchTab);
  const toggleReasoningViewMode = useTabsStore((state) => state.toggleReasoningViewMode);

  // Conversation settings store
  const { getSettings, saveSettings } = useConversationSettingsStore();

  // Subscribe to conversation settings for the active conversation
  const conversationSettings = useConversationSettingsStore(
    (state) => {
      const conversationId = activeTab?.conversationId;
      if (!conversationId) return null;
      return state.conversationSettings[conversationId]?.settings ?? null;
    }
  );

  // Helper function to convert tool input_schema to arguments
  const convertInputSchemaToArguments = useCallback((schema?: MCPTool['input_schema']): MCPPromptArgument[] => {
    if (!schema || !schema.properties) {
      return [];
    }

    const properties = schema.properties;
    const required = schema.required || [];

    return Object.entries(properties).map(([name, prop]) => ({
      name,
      description: prop.description,
      required: required.includes(name),
    }));
  }, []);

  // Get current settings for active tab
  // For new conversations (no conversationId), use tab.settings
  // For saved conversations, use the subscribed conversationSettings
  const currentSettings = useMemo(() => {
    if (activeTab?.settings) {
      return activeTab.settings;
    }
    if (conversationSettings) {
      return conversationSettings;
    }
    return getSettings(activeTab?.conversationId ?? null);
  }, [activeTab?.conversationId, activeTab?.settings, conversationSettings, getSettings]);

  // Update settings for active tab
  const handleUpdateSettings = useCallback(
    (updates: Partial<typeof currentSettings>) => {
      // Get fresh tab data to avoid stale closures
      const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
      if (!tab) return;

      // Get current settings from the appropriate source
      let currentSettings: ChatSettings;
      if (tab.settings) {
        // New conversation - use tab.settings
        currentSettings = tab.settings;
      } else if (tab.conversationId) {
        // Saved conversation - get from store
        const storedSettings = useConversationSettingsStore.getState()
          .conversationSettings[tab.conversationId]?.settings;
        currentSettings = storedSettings || getSettings(tab.conversationId);
      } else {
        // Fallback to defaults
        currentSettings = getSettings(null);
      }

      const newSettings = { ...currentSettings, ...updates };

      if (tab.conversationId) {
        // Save to conversation settings
        saveSettings(tab.conversationId, newSettings);
      } else {
        // Save to tab settings (will be migrated when conversation gets ID)
        updateTab(tab.id, { settings: newSettings });
      }
    },
    [activeTabId, getSettings, saveSettings, updateTab]
  );

  // When new conversation gets an ID, migrate tab settings to conversation settings
  useEffect(() => {
    if (activeTab?.conversationId && activeTab.settings) {
      saveSettings(activeTab.conversationId, activeTab.settings);
      updateTab(activeTab.id, { settings: null });
    }
  }, [activeTab?.conversationId, activeTab?.settings, activeTab?.id, saveSettings, updateTab]);

  // Conversation management
  const {
    conversations,
    isLoading: conversationsLoading,
    error: conversationsError,
    selectedConversationId,
    fetchConversations,
    deleteConversation,
    archiveConversation,
    updateConversationTitle,
    selectConversation,
  } = useConversations(client);

  // View filter and search query from store
  const viewFilter = useViewFilter();
  const searchQuery = useSearchQuery();
  const setViewFilter = useConversationsStore((state) => state.setViewFilter);
  const setSearchQuery = useConversationsStore((state) => state.setSearchQuery);

  // Filter conversations based on active/archived view
  const filteredConversations = useMemo(() => {
    return conversations.filter((conv) => {
      const isArchived = conv.archived_at !== null;
      const isActive = !isArchived;

      if (viewFilter === 'active') {
        return isActive;
      } else {
        return isArchived;
      }
    });
  }, [conversations, viewFilter]);

  // Process search results into conversation list
  const searchResultConversations = useMemo(
    () => processSearchResults(searchResults, conversations),
    [searchResults, conversations],
  );

  // Display conversations: search results when searching, filtered list otherwise
  const displayConversations = searchResults ? searchResultConversations || [] : filteredConversations;

  // Clear search results when search query is cleared
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setSearchResults(null);
    }
  }, [searchQuery]);

  // Load conversation history
  const {
    isLoading: isLoadingHistory,
    loadConversation,
  } = useLoadConversation(client);

  // Build messages array for display from active tab
  const messages = useMemo(() => {
    if (!activeTab) return [];

    const msgs = [...activeTab.messages];

    // Add current streaming message if tab is streaming
    if (activeTab.isStreaming) {
      msgs.push({
        role: 'assistant',
        content: activeTab.streamingContent || '',
        reasoningEvents: activeTab.reasoningEvents,
        usage: activeTab.usage,
      });
    }

    return msgs;
  }, [
    activeTab?.messages,
    activeTab?.isStreaming,
    activeTab?.streamingContent,
    activeTab?.reasoningEvents,
    activeTab?.usage,
  ]);

  const handleSendMessage = async (userMessage: string) => {
    if (!activeTab) return;

    // Determine temperature:
    // - GPT-5 models require temp=1
    // - Claude models with reasoning_effort require temp=1
    const isGPT5Model = currentSettings.model.toLowerCase().startsWith('gpt-5');
    const hasReasoningEffort = currentSettings.reasoningEffort !== undefined;
    const temperature = (isGPT5Model || hasReasoningEffort) ? 1.0 : currentSettings.temperature;

    // Send to API with current settings (hook handles adding user message to tab)
    const conversationId = await sendMessageForTab(activeTab.id, userMessage, {
      model: currentSettings.model,
      routingMode: currentSettings.routingMode,
      temperature: temperature,
      systemPrompt: currentSettings.systemPrompt || undefined,
      reasoningEffort: currentSettings.reasoningEffort,
    });

    if (conversationId) {
      if (!activeTab.conversationId) {
        updateTab(activeTab.id, { conversationId });
      }
      fetchConversations();
      selectConversation(conversationId);
    }
  };

  const handleCancel = () => {
    if (!activeTab) return;

    // Cancel stream (hook handles state cleanup)
    cancelStreamForTab(activeTab.id);

    // If there was partial content, save it as a cancelled message
    if (activeTab.streamingContent) {
      const updatedMessages = [
        ...activeTab.messages,
        {
          role: 'assistant' as const,
          content: activeTab.streamingContent + ' [cancelled]',
          reasoningEvents: activeTab.reasoningEvents,
          usage: activeTab.usage,
        },
      ];

      updateTab(activeTab.id, {
        messages: updatedMessages,
      });
    }

    // Auto-focus chat input after canceling
    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  };

  // Auto-focus chat input after streaming completes
  useEffect(() => {
    if (!activeTab) return;

    // Focus input when streaming stops
    if (!activeTab.isStreaming) {
      setTimeout(() => {
        chatLayoutRef.current?.focusInput();
      }, 0);
    }
  }, [activeTab?.id, activeTab?.isStreaming]);

  // Handle conversation selection from sidebar
  const handleSelectConversation = useCallback(
    async (id: string) => {
      selectConversation(id);

      const existingTab = findTabByConversationId(id);

      if (existingTab) {
        switchTab(existingTab.id);
      } else {
        try {
          const history = await loadConversation(id);

          const conversation = conversations.find((c) => c.id === id);
          const title = conversation?.title || 'Untitled';

          addTab({
            conversationId: id,
            title: title,
            messages: history,
            input: '',
            isStreaming: false,
            streamingContent: '',
            reasoningEvents: [],
            usage: null,
            streamError: null,
            settings: null,
            reasoningViewMode: 'text',
          });
        } catch (err) {
          console.error('Failed to load conversation:', err);
        }
      }
    },
    [
      selectConversation,
      findTabByConversationId,
      switchTab,
      loadConversation,
      conversations,
      addTab,
    ],
  );

  const handleNewConversation = useCallback(() => {
    addTab({
      conversationId: null,
      title: 'New Chat',
      messages: [],
      input: '',
      isStreaming: false,
      streamingContent: '',
      reasoningEvents: [],
      usage: null,
      streamError: null,
      settings: null,
      reasoningViewMode: 'text',
    });
    selectConversation(null);

    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  }, [addTab, selectConversation]);

  // Handle new tab button
  const handleNewTab = useCallback(() => {
    handleNewConversation();
  }, [handleNewConversation]);

  // Handle delete conversation
  const handleDeleteConversation = useCallback(
    async (id: string) => {
      await deleteConversation(id);

      // If conversation is open in a tab, close that tab
      const tabToClose = findTabByConversationId(id);
      if (tabToClose) {
        useTabsStore.getState().removeTab(tabToClose.id);
      }
    },
    [deleteConversation, findTabByConversationId],
  );

  // Handle archive conversation
  const handleArchiveConversation = useCallback(
    async (id: string) => {
      await archiveConversation(id);

      // If conversation is open in a tab, close that tab
      const tabToClose = findTabByConversationId(id);
      if (tabToClose) {
        useTabsStore.getState().removeTab(tabToClose.id);
      }
    },
    [archiveConversation, findTabByConversationId],
  );

  // Handle delete message (and all subsequent messages)
  const handleDeleteMessage = useCallback(
    async (messageIndex: number) => {
      // Get fresh tab data to avoid stale closures
      const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
      if (!tab?.conversationId) return;

      const message = tab.messages[messageIndex];
      if (!message?.sequenceNumber) {
        console.warn('Cannot delete message without sequence number');
        return;
      }

      try {
        // Delete from API
        await client.deleteMessage(tab.conversationId, message.sequenceNumber);

        // Update local state: remove this message and all after it
        const updatedMessages = tab.messages.slice(0, messageIndex);
        updateTab(tab.id, { messages: updatedMessages });

        // Refresh conversation list to update message counts
        await fetchConversations();
      } catch (err) {
        console.error('Failed to delete message:', err);
        toast.error('Failed to delete message');
      }
    },
    [activeTabId, client, updateTab, fetchConversations, toast],
  );

  // Handle regenerate message (delete assistant message and generate new response)
  const handleRegenerateMessage = useCallback(
    async (messageIndex: number) => {
      // Get fresh tab data to avoid stale closures
      const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
      if (!tab?.conversationId) return;

      const message = tab.messages[messageIndex];
      if (!message?.sequenceNumber) {
        console.warn('Cannot regenerate message without sequence number');
        return;
      }

      try {
        // Delete the assistant message (and all after it) from API
        await client.deleteMessage(tab.conversationId, message.sequenceNumber);

        // Update local state: remove this message and all after it
        const updatedMessages = tab.messages.slice(0, messageIndex);
        updateTab(tab.id, { messages: updatedMessages });

        // Get current settings (handles both new conversations and saved ones)
        const settings = tab.settings || getSettings(tab.conversationId);

        // Determine temperature:
        // - GPT-5 models require temp=1
        // - Claude models with reasoning_effort require temp=1
        const isGPT5Model = settings.model.toLowerCase().startsWith('gpt-5');
        const hasReasoningEffort = settings.reasoningEffort !== undefined;
        const temperature = (isGPT5Model || hasReasoningEffort) ? 1.0 : settings.temperature;

        // Use hook's regenerate method
        await regenerateForTab(tab.id, {
          model: settings.model,
          routingMode: settings.routingMode,
          temperature: temperature,
          systemPrompt: settings.systemPrompt || undefined,
          reasoningEffort: settings.reasoningEffort,
        });

        // Refresh conversations list
        await fetchConversations();
      } catch (err) {
        console.error('Failed to regenerate message:', err);
        toast.error('Failed to regenerate message');
      }
    },
    [activeTabId, client, updateTab, getSettings, regenerateForTab, fetchConversations, toast],
  );

  // Handle branch conversation (create new conversation from this point)
  const handleBranchConversation = useCallback(
    async (messageIndex: number) => {
      // Get fresh tab data to avoid stale closures
      const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
      if (!tab?.conversationId) return;

      const message = tab.messages[messageIndex];
      if (!message?.sequenceNumber) {
        console.warn('Cannot branch from message without sequence number');
        return;
      }

      try {
        // Create branched conversation via API
        const branchedConversation = await client.branchConversation(
          tab.conversationId,
          message.sequenceNumber,
        );

        // Refresh conversation list to show new conversation
        await fetchConversations();

        // Open branched conversation in a new tab
        const history = branchedConversation.messages.map((msg) => {
          let usage: Usage | undefined = undefined;
          if (msg.metadata?.usage) {
            usage = {
              ...(msg.metadata.usage as Usage),
              ...(msg.metadata.cost && {
                prompt_cost: msg.metadata.cost.prompt_cost,
                completion_cost: msg.metadata.cost.completion_cost,
                total_cost: msg.metadata.cost.total_cost,
              }),
            };
          }

          return {
            id: msg.id,
            sequenceNumber: msg.sequence_number,
            role: msg.role as 'user' | 'assistant' | 'system',
            content: msg.content || '',
            reasoningEvents: msg.reasoning_events
              ? (msg.reasoning_events as unknown as ReasoningEvent[])
              : undefined,
            usage,
          };
        });

        addTab({
          conversationId: branchedConversation.id,
          title: branchedConversation.title || 'Branched Conversation',
          messages: history,
          input: '',
          isStreaming: false,
          streamingContent: '',
          reasoningEvents: [],
          usage: null,
          streamError: null,
          settings: null,
          reasoningViewMode: 'text',
        });

        // Select the newly branched conversation in the sidebar
        selectConversation(branchedConversation.id);

        toast.success('Conversation branched successfully');
      } catch (err) {
        console.error('Failed to branch conversation:', err);
        toast.error('Failed to branch conversation');
      }
    },
    [activeTabId, client, fetchConversations, addTab, selectConversation, toast],
  );

  // Handle search
  const handleSearch = useCallback(
    async (query: string) => {
      try {
        const results = await client.searchMessages(query, {
          archiveFilter: viewFilter === 'archived' ? 'archived' : 'active',
          limit: 20,
        });
        setSearchResults(results.results);
      } catch (err) {
        console.error('Search failed:', err);
        setSearchResults(null);
      }
    },
    [client, viewFilter],
  );

  // Update tab input when it changes
  const handleInputChange = useCallback(
    (value: string) => {
      if (activeTabId) {
        updateTab(activeTabId, { input: value });
      }
    },
    [activeTabId, updateTab],
  );

  // Handle close current tab
  const handleCloseCurrentTab = useCallback(() => {
    if (!activeTabId || tabs.length === 1) {
      // Don't close if it's the only tab
      return;
    }
    useTabsStore.getState().removeTab(activeTabId);
  }, [activeTabId, tabs.length]);

  // Handle toggle settings (global state)
  const handleToggleSettings = useCallback(() => {
    setIsSettingsOpen(!isSettingsOpen);
  }, [isSettingsOpen]);

  // Handle toggle conversations sidebar
  const handleToggleConversations = useCallback(() => {
    setIsConversationsOpen(!isConversationsOpen);
  }, [isConversationsOpen]);

  // Handle navigate to previous tab (with wrapping)
  const handlePreviousTab = useCallback(() => {
    if (tabs.length === 0) return;

    const currentIndex = tabs.findIndex((tab) => tab.id === activeTabId);

    // Navigate to previous tab (wrap to end if at beginning)
    const prevIndex = currentIndex === 0 ? tabs.length - 1 : currentIndex - 1;
    switchTab(tabs[prevIndex].id);

    // Focus chat input after switching
    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  }, [tabs, activeTabId, switchTab]);

  // Handle navigate to next tab (with wrapping)
  const handleNextTab = useCallback(() => {
    if (tabs.length === 0) return;

    const currentIndex = tabs.findIndex((tab) => tab.id === activeTabId);

    // Navigate to next tab (wrap to beginning if at end)
    const nextIndex = currentIndex === tabs.length - 1 ? 0 : currentIndex + 1;
    switchTab(tabs[nextIndex].id);

    // Focus chat input after switching
    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  }, [tabs, activeTabId, switchTab]);

  // Handle prompt selection from command palette
  const handleSelectPrompt = useCallback(
    async (prompt: MCPPrompt) => {
      if (!activeTabId) return;

      // Check if prompt has arguments
      if (prompt.arguments && prompt.arguments.length > 0) {
        // Show arguments dialog
        setSelectedPrompt(prompt);
        setSelectedTool(null);
        setSelectedItemType('prompt');
        setIsArgumentsDialogOpen(true);
      } else {
        // No arguments - execute immediately
        try {
          const result = await client.executeMCPPrompt(prompt.name, {});

          // Extract prompt content from last message
          const promptContent = result.messages.length > 0
            ? result.messages[result.messages.length - 1].content
            : prompt.description || prompt.name;

          // Insert into input
          const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
          const currentInput = tab?.input || '';
          const newInput = currentInput ? `${currentInput}\n\n${promptContent}` : promptContent;

          updateTab(activeTabId, { input: newInput });

          // Focus chat input after inserting prompt
          setTimeout(() => {
            chatLayoutRef.current?.focusInput();
          }, 0);
        } catch (err) {
          console.error('Failed to execute prompt:', err);
          toast.error('Failed to execute prompt');
        }
      }
    },
    [activeTabId, updateTab, client, toast],
  );

  // Handle tool execution
  const handleToolExecution = useCallback(
    async (tool: MCPTool, args: Record<string, unknown>) => {
      setIsExecuting(true);

      try {
        const result = await client.executeMCPTool(tool.name, args);

        // Store result and show result dialog
        setToolExecutionResult(result);
        setIsToolResultDialogOpen(true);
        setIsArgumentsDialogOpen(false);
        setSelectedTool(null);

        if (!result.success) {
          toast.error(`Tool execution failed`);
        }
      } catch (err) {
        console.error('Failed to execute tool:', err);
        toast.error(`Failed to execute tool: ${err instanceof Error ? err.message : 'Unknown error'}`);
      } finally {
        setIsExecuting(false);
      }
    },
    [client, toast],
  );

  // Handle tool selection from command palette
  const handleSelectTool = useCallback(
    (tool: MCPTool) => {
      const toolArguments = convertInputSchemaToArguments(tool.input_schema);

      if (toolArguments.length > 0) {
        // Show arguments dialog
        setSelectedTool(tool);
        setSelectedPrompt(null);
        setSelectedItemType('tool');
        setIsArgumentsDialogOpen(true);
      } else {
        // No arguments - execute immediately
        setSelectedTool(tool);
        setSelectedPrompt(null);
        setSelectedItemType('tool');
        // Execute with empty args
        handleToolExecution(tool, {});
      }
    },
    [convertInputSchemaToArguments, handleToolExecution],
  );

  // Handle arguments dialog submission
  const handleArgumentsSubmit = useCallback(
    async (args: Record<string, unknown>) => {
      if (!activeTabId) return;

      if (selectedItemType === 'prompt' && selectedPrompt) {
        setIsExecuting(true);
        try {
          // Execute prompt with arguments (prompts use string values)
          const stringArgs = Object.fromEntries(
            Object.entries(args).map(([k, v]) => [k, String(v ?? '')]),
          );
          const result = await client.executeMCPPrompt(selectedPrompt.name, stringArgs);

          // Extract prompt content from last message
          const promptContent = result.messages.length > 0
            ? result.messages[result.messages.length - 1].content
            : selectedPrompt.description || selectedPrompt.name;

          // Insert into input
          const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
          const currentInput = tab?.input || '';
          const newInput = currentInput ? `${currentInput}\n\n${promptContent}` : promptContent;

          updateTab(activeTabId, { input: newInput });

          // Close dialog and reset state
          setIsArgumentsDialogOpen(false);
          setSelectedPrompt(null);

          // Focus chat input after inserting prompt
          setTimeout(() => {
            chatLayoutRef.current?.focusInput();
          }, 0);
        } catch (err) {
          console.error('Failed to execute prompt with arguments:', err);
          toast.error('Failed to execute prompt');
        } finally {
          setIsExecuting(false);
        }
      } else if (selectedItemType === 'tool' && selectedTool) {
        // Execute tool with arguments
        await handleToolExecution(selectedTool, args);
      }
    },
    [selectedPrompt, selectedTool, selectedItemType, activeTabId, client, updateTab, toast, handleToolExecution],
  );

  // Handle copy tool result to clipboard
  const handleCopyToolResult = useCallback(() => {
    if (!toolExecutionResult) return;

    const formatResult = (result: unknown): string => {
      if (typeof result === 'string') {
        return result;
      }
      try {
        return JSON.stringify(result, null, 2);
      } catch {
        return String(result);
      }
    };

    const formattedResult = formatResult(toolExecutionResult.result);

    navigator.clipboard.writeText(formattedResult)
      .catch((err) => {
        console.error('Failed to copy to clipboard:', err);
        toast.error('Failed to copy to clipboard');
      });
  }, [toolExecutionResult, toast]);

  // Handle send tool result to chat
  const handleSendToolResultToChat = useCallback(() => {
    if (!toolExecutionResult || !activeTabId) return;

    const formatResult = (result: unknown): string => {
      if (typeof result === 'string') {
        return result;
      }
      try {
        return JSON.stringify(result, null, 2);
      } catch {
        return String(result);
      }
    };

    const formattedResult = formatResult(toolExecutionResult.result);

    // Insert into input
    const tab = useTabsStore.getState().tabs.find((t) => t.id === activeTabId);
    const currentInput = tab?.input || '';
    const newInput = currentInput ? `${currentInput}\n\n${formattedResult}` : formattedResult;

    updateTab(activeTabId, { input: newInput });

    // Close dialog and reset state
    setIsToolResultDialogOpen(false);
    setToolExecutionResult(null);

    // Focus chat input after inserting result
    setTimeout(() => {
      chatLayoutRef.current?.focusInput();
    }, 0);
  }, [toolExecutionResult, activeTabId, updateTab]);

  // Keyboard shortcuts
  useKeyboardShortcuts([
    // Cmd+N (Ctrl+N on Windows/Linux): New conversation
    createCrossPlatformShortcut('n', handleNewConversation),

    // Cmd+W (Ctrl+W on Windows/Linux): Close current tab (unless only 1 tab)
    createCrossPlatformShortcut('w', handleCloseCurrentTab),

    // Cmd+, (Ctrl+, on Windows/Linux): Toggle settings panel
    createCrossPlatformShortcut(',', handleToggleSettings),

    // Cmd+/ (Ctrl+/ on Windows/Linux): Show keyboard shortcuts overlay
    createCrossPlatformShortcut('/', () => setIsShortcutsOverlayOpen(!isShortcutsOverlayOpen)),

    // Cmd+Shift+P (Ctrl+Shift+P on Windows/Linux): Open command palette
    createCrossPlatformShortcut('p', () => setIsCommandPaletteOpen(true), {
      shift: true,
    }),

    // Cmd+Shift+D (Ctrl+Shift+D on Windows/Linux): Toggle reasoning view mode
    createCrossPlatformShortcut('d', () => {
      if (activeTabId) {
        toggleReasoningViewMode(activeTabId);
      }
    }, {
      shift: true,
    }),

    // Cmd+Shift+F (Ctrl+Shift+F on Windows/Linux): Focus search box
    createCrossPlatformShortcut('f', () => conversationListRef.current?.focusSearch(), {
      shift: true,
    }),

    // Cmd+Shift+[ (Ctrl+Shift+[ on Windows/Linux): Previous tab
    createCrossPlatformShortcut('[', handlePreviousTab, {
      shift: true,
    }),

    // Cmd+Shift+] (Ctrl+Shift+] on Windows/Linux): Next tab
    createCrossPlatformShortcut(']', handleNextTab, {
      shift: true,
    }),

    // Escape: Focus chat input (works even when focused in search/input fields)
    // Note: When shortcuts overlay is open, it handles Escape itself
    {
      key: 'Escape',
      handler: () => {
        // Don't handle if shortcuts overlay is open (it will handle Escape)
        if (!isShortcutsOverlayOpen) {
          chatLayoutRef.current?.focusInput();
        }
      },
      preventDefault: true,
      allowInInputFields: true,
    },
  ]);

  // Sync conversation titles to tabs when they change
  useEffect(() => {
    tabs.forEach((tab) => {
      if (tab.conversationId) {
        const conversation = conversations.find((c) => c.id === tab.conversationId);
        if (conversation) {
          const newTitle = conversation.title || 'Untitled';
          if (tab.title !== newTitle) {
            updateTab(tab.id, { title: newTitle });
          }
        }
      }
    });
  }, [conversations, tabs, updateTab]);

  return (
    <>
      <AppLayout
        conversationsSidebar={
          <ConversationList
            ref={conversationListRef}
            conversations={displayConversations}
            selectedConversationId={selectedConversationId}
            isLoading={conversationsLoading}
            error={conversationsError}
            viewFilter={viewFilter}
            searchQuery={searchQuery}
            onSelectConversation={handleSelectConversation}
            onNewConversation={handleNewConversation}
            onDeleteConversation={handleDeleteConversation}
            onArchiveConversation={handleArchiveConversation}
            onUpdateTitle={updateConversationTitle}
            onRefresh={fetchConversations}
            onViewFilterChange={setViewFilter}
            onSearchQueryChange={setSearchQuery}
            onSearch={handleSearch}
          />
        }
        isConversationsOpen={isConversationsOpen}
        tabBar={
          <TabBar
            onNewTab={handleNewTab}
            isSettingsOpen={isSettingsOpen}
            onToggleSettings={handleToggleSettings}
            isConversationsOpen={isConversationsOpen}
            onToggleConversations={handleToggleConversations}
          />
        }
      >
        <ChatLayout
          ref={chatLayoutRef}
          messages={messages}
          isStreaming={!!activeTab?.isStreaming}
          isLoadingHistory={isLoadingHistory}
          input={input}
          onInputChange={handleInputChange}
          onSendMessage={handleSendMessage}
          onCancel={handleCancel}
          isSettingsOpen={isSettingsOpen}
          reasoningViewMode={activeTab?.reasoningViewMode || 'text'}
          settingsPanel={
            <SettingsPanel
              availableModels={models}
              isLoadingModels={isLoadingModels}
              settings={currentSettings}
              onUpdateSettings={handleUpdateSettings}
            />
          }
          onDeleteMessage={handleDeleteMessage}
          onRegenerateMessage={handleRegenerateMessage}
          onBranchConversation={handleBranchConversation}
        />
      </AppLayout>

      <KeyboardShortcutsOverlay
        isOpen={isShortcutsOverlayOpen}
        onClose={() => setIsShortcutsOverlayOpen(false)}
      />

      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        prompts={prompts}
        tools={tools}
        onSelectPrompt={handleSelectPrompt}
        onSelectTool={handleSelectTool}
        isLoading={isLoadingPrompts || isLoadingTools}
      />

      <ArgumentsDialog
        isOpen={isArgumentsDialogOpen}
        onClose={() => {
          setIsArgumentsDialogOpen(false);
          setSelectedPrompt(null);
          setSelectedTool(null);
        }}
        type={selectedItemType}
        name={
          selectedItemType === 'prompt'
            ? selectedPrompt?.name || ''
            : selectedTool?.name || ''
        }
        description={
          selectedItemType === 'prompt'
            ? selectedPrompt?.description
            : selectedTool?.description
        }
        arguments={
          selectedItemType === 'prompt'
            ? selectedPrompt?.arguments || []
            : convertInputSchemaToArguments(selectedTool?.input_schema)
        }
        inputSchema={selectedItemType === 'tool' ? selectedTool?.input_schema : undefined}
        onSubmit={handleArgumentsSubmit}
        isExecuting={isExecuting}
      />

      <ToolExecutionDialog
        isOpen={isToolResultDialogOpen}
        onClose={() => {
          setIsToolResultDialogOpen(false);
          setToolExecutionResult(null);
        }}
        result={toolExecutionResult}
        onCopy={handleCopyToolResult}
        onSendToChat={handleSendToolResultToChat}
      />
    </>
  );
}

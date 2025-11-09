/**
 * SettingsPanel component - main settings sidebar.
 *
 * Contains all chat configuration options:
 * - Model selection
 * - Routing mode
 * - Temperature
 * - System prompt
 *
 * Uses Zustand store for state management (no prop drilling).
 */

import { RoutingModeSelector } from './RoutingModeSelector';
import { Textarea } from '../ui/textarea';
import type { ChatSettings } from '../../store/chat-store';
import type { ModelInfo } from '../../types/openai';

export interface SettingsPanelProps {
  availableModels: ModelInfo[];
  isLoadingModels: boolean;
  settings: ChatSettings;
  onUpdateSettings: (settings: Partial<ChatSettings>) => void;
}

/**
 * Settings sidebar panel with all configuration options.
 *
 * @example
 * ```tsx
 * <SettingsPanel
 *   availableModels={models}
 *   isLoadingModels={false}
 *   settings={settings}
 *   onUpdateSettings={updateSettings}
 * />
 * ```
 */
export function SettingsPanel({
  availableModels,
  isLoadingModels,
  settings,
  onUpdateSettings: updateSettings,
}: SettingsPanelProps): JSX.Element {

  // Check if current model is gpt-5 series (requires temp=1)
  const isGPT5Model = settings.model.toLowerCase().startsWith('gpt-5');

  // Check if current model supports reasoning
  const currentModelInfo = availableModels.find((m) => m.id === settings.model);
  const supportsReasoning = currentModelInfo?.supports_reasoning ?? false;

  // Reasoning effort only applies to passthrough/auto modes (not custom "Reasoning" mode)
  const showReasoningEffort = supportsReasoning &&
    settings.routingMode !== 'reasoning';

  // Temperature is locked to 1.0 for GPT-5 models or when reasoning effort is set
  const hasReasoningEffort = settings.reasoningEffort !== undefined;
  const isTemperatureLocked = isGPT5Model || hasReasoningEffort;

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="border-b pb-3">
        <h2 className="text-lg font-semibold text-foreground">Settings</h2>
        <p className="text-xs text-muted-foreground mt-1">
          Configure your chat experience
        </p>
      </div>

      {/* Routing Mode */}
      <RoutingModeSelector
        value={settings.routingMode}
        onChange={(mode) => updateSettings({ routingMode: mode })}
      />

      {/* Model Selection */}
      <div className="space-y-2">
        <label htmlFor="model-select" className="text-sm font-medium text-foreground">
          Model
        </label>
        {isLoadingModels ? (
          <div className="text-xs text-muted-foreground">Loading models...</div>
        ) : (
          <select
            id="model-select"
            value={settings.model}
            onChange={(e) => updateSettings({ model: e.target.value })}
            className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background focus:ring-2 focus:ring-ring focus:border-transparent"
          >
            {availableModels.length === 0 ? (
              <option value={settings.model}>{settings.model} (default)</option>
            ) : (
              availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.id}
                </option>
              ))
            )}
          </select>
        )}
        <p className="text-xs text-muted-foreground">
          {availableModels.length} model{availableModels.length !== 1 ? 's' : ''} available
        </p>
      </div>

      {/* Temperature */}
      <div className="space-y-2">
        <label htmlFor="temperature-slider" className="text-sm font-medium text-foreground">
          Temperature: {isTemperatureLocked ? '1.0' : settings.temperature.toFixed(1)}
        </label>
        {isTemperatureLocked ? (
          <div className="px-3 py-2 text-sm bg-muted rounded-md text-muted-foreground">
            {isGPT5Model
              ? `Fixed at 1.0 (required for ${settings.model})`
              : 'Fixed at 1.0 (required for reasoning effort)'}
          </div>
        ) : (
          <>
            <input
              id="temperature-slider"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={settings.temperature}
              onChange={(e) => updateSettings({ temperature: parseFloat(e.target.value) })}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Deterministic</span>
              <span>Balanced</span>
              <span>Varied</span>
            </div>
          </>
        )}
      </div>

      {/* Reasoning Effort - only show for models that support it and not in "Reasoning" mode */}
      {showReasoningEffort && (
        <div className="space-y-2">
          <label htmlFor="reasoning-effort" className="text-sm font-medium text-foreground">
            Model Reasoning Effort
          </label>
          <select
            id="reasoning-effort"
            value={settings.reasoningEffort || ''}
            onChange={(e) => updateSettings({
              reasoningEffort: e.target.value ? e.target.value as 'low' | 'medium' | 'high' : undefined
            })}
            className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background focus:ring-2 focus:ring-ring focus:border-transparent"
          >
            <option value="">Default</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          <p className="text-xs text-muted-foreground">
            Controls the model's internal reasoning depth
          </p>
        </div>
      )}

      {/* System Prompt */}
      <div className="space-y-2">
        <label htmlFor="system-prompt" className="text-sm font-medium text-foreground">
          System Prompt
        </label>
        <Textarea
          id="system-prompt"
          value={settings.systemPrompt}
          onChange={(e) => updateSettings({ systemPrompt: e.target.value })}
          placeholder="You are a helpful assistant..."
          className="min-h-[100px] text-xs resize-none"
        />
        <p className="text-xs text-muted-foreground">
          Optional instructions for the AI
        </p>
      </div>
    </div>
  );
}

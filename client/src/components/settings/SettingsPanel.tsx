/**
 * SettingsPanel component - main settings sidebar.
 *
 * Contains all chat configuration options:
 * - Model selection
 * - Routing mode
 * - Temperature & max tokens
 * - System prompt
 */

import { RoutingModeSelector } from './RoutingModeSelector';
import { Textarea } from '../ui/textarea';
import type { RoutingModeType } from '../../constants';

export interface ChatSettings {
  model: string;
  routingMode: RoutingModeType;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
}

export interface SettingsPanelProps {
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
  availableModels: string[];
  isLoadingModels: boolean;
}

/**
 * Settings sidebar panel with all configuration options.
 *
 * @example
 * ```tsx
 * const [settings, setSettings] = useState<ChatSettings>({...});
 * <SettingsPanel
 *   settings={settings}
 *   onSettingsChange={setSettings}
 *   availableModels={models}
 * />
 * ```
 */
export function SettingsPanel({
  settings,
  onSettingsChange,
  availableModels,
  isLoadingModels,
}: SettingsPanelProps): JSX.Element {
  const updateSetting = <K extends keyof ChatSettings>(key: K, value: ChatSettings[K]) => {
    onSettingsChange({ ...settings, [key]: value });
  };

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
        onChange={(mode) => updateSetting('routingMode', mode)}
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
            onChange={(e) => updateSetting('model', e.target.value)}
            className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background focus:ring-2 focus:ring-ring focus:border-transparent"
          >
            {availableModels.length === 0 ? (
              <option value={settings.model}>{settings.model} (default)</option>
            ) : (
              availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
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
          Temperature: {settings.temperature.toFixed(1)}
        </label>
        <input
          id="temperature-slider"
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={settings.temperature}
          onChange={(e) => updateSetting('temperature', parseFloat(e.target.value))}
          className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Focused</span>
          <span>Balanced</span>
          <span>Creative</span>
        </div>
      </div>

      {/* Max Tokens */}
      <div className="space-y-2">
        <label htmlFor="max-tokens-slider" className="text-sm font-medium text-foreground">
          Max Tokens: {settings.maxTokens}
        </label>
        <input
          id="max-tokens-slider"
          type="range"
          min="256"
          max="4096"
          step="256"
          value={settings.maxTokens}
          onChange={(e) => updateSetting('maxTokens', parseInt(e.target.value))}
          className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>256</span>
          <span>2048</span>
          <span>4096</span>
        </div>
      </div>

      {/* System Prompt */}
      <div className="space-y-2">
        <label htmlFor="system-prompt" className="text-sm font-medium text-foreground">
          System Prompt
        </label>
        <Textarea
          id="system-prompt"
          value={settings.systemPrompt}
          onChange={(e) => updateSetting('systemPrompt', e.target.value)}
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

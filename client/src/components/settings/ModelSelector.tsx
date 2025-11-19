/**
 * ModelSelector component - rich model selection dropdown.
 *
 * Displays models with detailed information:
 * - Model name (primary line)
 * - Provider, pricing (secondary line)
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '../ui/select';
import type { ModelInfo } from '../../types/openai';

export interface ModelSelectorProps {
  models: ModelInfo[];
  value: string;
  onChange: (modelId: string) => void;
  isLoading?: boolean;
}

/**
 * Format pricing per million tokens.
 *
 * @param costPerToken - Cost per token (e.g., 0.0000025)
 * @returns Formatted price (e.g., "$2.50")
 */
function formatPrice(costPerToken: number): string {
  const perMillion = costPerToken * 1_000_000;
  return `$${perMillion.toFixed(2)}`;
}

/**
 * Get provider display name with proper capitalization.
 */
function formatProvider(ownedBy: string): string {
  // Special case mappings for non-standard capitalization
  const providerMap: Record<string, string> = {
    openai: 'OpenAI',
  };

  const normalized = ownedBy.toLowerCase();
  if (providerMap[normalized]) {
    return providerMap[normalized];
  }

  // Capitalize first letter for other providers
  return ownedBy.charAt(0).toUpperCase() + ownedBy.slice(1).toLowerCase();
}

/**
 * Model selector with rich display.
 *
 * @example
 * ```tsx
 * <ModelSelector
 *   models={availableModels}
 *   value={selectedModel}
 *   onChange={setSelectedModel}
 *   isLoading={false}
 * />
 * ```
 */
export function ModelSelector({
  models,
  value,
  onChange,
  isLoading = false,
}: ModelSelectorProps): JSX.Element {
  if (isLoading) {
    return (
      <div className="text-xs text-muted-foreground">Loading models...</div>
    );
  }

  // If no models available, show fallback
  if (models.length === 0) {
    return (
      <div className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background">
        {value} (default)
      </div>
    );
  }

  // Find current model for display
  const currentModel = models.find((m) => m.id === value);
  const currentProvider = currentModel ? formatProvider(currentModel.owned_by) : '';
  const currentInputPrice = currentModel ? formatPrice(currentModel.input_cost_per_token) : '';
  const currentOutputPrice = currentModel ? formatPrice(currentModel.output_cost_per_token) : '';

  // Sort models by provider, then by input cost (cheapest first)
  const sortedModels = [...models].sort((a, b) => {
    // First sort by provider (case-insensitive)
    const providerA = a.owned_by.toLowerCase();
    const providerB = b.owned_by.toLowerCase();
    if (providerA !== providerB) {
      return providerA.localeCompare(providerB);
    }
    // Then sort by input cost (ascending - cheapest first)
    return a.input_cost_per_token - b.input_cost_per_token;
  });

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-full">
        <div className="flex flex-col gap-1.5 text-left">
          <div className="font-medium leading-none">{value}</div>
          {currentModel && (
            <div className="text-xs text-muted-foreground leading-none">
              {currentProvider} • In: {currentInputPrice} Out: {currentOutputPrice}
            </div>
          )}
        </div>
      </SelectTrigger>
      <SelectContent>
        {sortedModels.map((model) => {
          const inputPrice = formatPrice(model.input_cost_per_token);
          const outputPrice = formatPrice(model.output_cost_per_token);
          const provider = formatProvider(model.owned_by);

          return (
            <SelectItem key={model.id} value={model.id}>
              <div className="flex flex-col gap-1.5">
                <div className="font-medium leading-none">{model.id}</div>
                <div className="text-xs text-muted-foreground leading-none">
                  {provider} • In: {inputPrice} Out: {outputPrice}
                </div>
              </div>
            </SelectItem>
          );
        })}
      </SelectContent>
    </Select>
  );
}

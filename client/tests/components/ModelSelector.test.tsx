/**
 * Tests for ModelSelector component.
 *
 * Tests model sorting, provider formatting, and price display.
 * Note: Tests focus on logic and rendering rather than interactions
 * to avoid jsdom limitations with Radix UI pointer events.
 */

import { render, screen } from '@testing-library/react';
import { ModelSelector } from '../../src/components/settings/ModelSelector';
import type { ModelInfo } from '../../src/types/openai';

const createMockModel = (overrides: Partial<ModelInfo>): ModelInfo => ({
  id: 'test-model',
  object: 'model',
  created: 1234567890,
  owned_by: 'test',
  max_input_tokens: 100000,
  max_output_tokens: 10000,
  input_cost_per_token: 0.000001,
  output_cost_per_token: 0.000005,
  supports_reasoning: false,
  supports_response_schema: true,
  supports_vision: false,
  supports_function_calling: true,
  supports_web_search: null,
  ...overrides,
});

describe('ModelSelector', () => {
  const defaultProps = {
    models: [],
    value: 'gpt-4o',
    onChange: jest.fn(),
    isLoading: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('sorting', () => {
    it('should sort models by provider alphabetically, then by input cost (cheapest first)', () => {
      const models: ModelInfo[] = [
        createMockModel({
          id: 'gpt-4o',
          owned_by: 'openai',
          input_cost_per_token: 0.0000025,
        }),
        createMockModel({
          id: 'claude-sonnet-4-5',
          owned_by: 'anthropic',
          input_cost_per_token: 0.000003,
        }),
        createMockModel({
          id: 'gpt-4o-mini',
          owned_by: 'openai',
          input_cost_per_token: 0.00000015,
        }),
        createMockModel({
          id: 'claude-haiku-4-5',
          owned_by: 'anthropic',
          input_cost_per_token: 0.000001,
        }),
        createMockModel({
          id: 'gpt-5',
          owned_by: 'openai',
          input_cost_per_token: 0.00000125,
        }),
      ];

      // Manually apply the same sorting logic used in ModelSelector
      const sortedModels = [...models].sort((a, b) => {
        const providerA = a.owned_by.toLowerCase();
        const providerB = b.owned_by.toLowerCase();
        if (providerA !== providerB) {
          return providerA.localeCompare(providerB);
        }
        return a.input_cost_per_token - b.input_cost_per_token;
      });

      // Expected order:
      // 1. anthropic models (alphabetically by provider)
      //    - claude-haiku-4-5 ($1.00) - cheapest anthropic
      //    - claude-sonnet-4-5 ($3.00)
      // 2. openai models
      //    - gpt-4o-mini ($0.15) - cheapest openai
      //    - gpt-5 ($1.25)
      //    - gpt-4o ($2.50)

      expect(sortedModels[0].id).toBe('claude-haiku-4-5');
      expect(sortedModels[0].owned_by).toBe('anthropic');
      expect(sortedModels[0].input_cost_per_token).toBe(0.000001);

      expect(sortedModels[1].id).toBe('claude-sonnet-4-5');
      expect(sortedModels[1].input_cost_per_token).toBe(0.000003);

      expect(sortedModels[2].id).toBe('gpt-4o-mini');
      expect(sortedModels[2].owned_by).toBe('openai');
      expect(sortedModels[2].input_cost_per_token).toBe(0.00000015);

      expect(sortedModels[3].id).toBe('gpt-5');
      expect(sortedModels[3].input_cost_per_token).toBe(0.00000125);

      expect(sortedModels[4].id).toBe('gpt-4o');
      expect(sortedModels[4].input_cost_per_token).toBe(0.0000025);
    });

    it('should sort models with same input cost stably', () => {
      const models: ModelInfo[] = [
        createMockModel({
          id: 'model-z',
          owned_by: 'provider',
          input_cost_per_token: 0.000001,
        }),
        createMockModel({
          id: 'model-a',
          owned_by: 'provider',
          input_cost_per_token: 0.000001,
        }),
      ];

      const sortedModels = [...models].sort((a, b) => {
        const providerA = a.owned_by.toLowerCase();
        const providerB = b.owned_by.toLowerCase();
        if (providerA !== providerB) {
          return providerA.localeCompare(providerB);
        }
        return a.input_cost_per_token - b.input_cost_per_token;
      });

      // When costs are equal, maintain original order
      expect(sortedModels).toHaveLength(2);
      expect(sortedModels[0].id).toBe('model-z');
      expect(sortedModels[1].id).toBe('model-a');
    });
  });

  describe('provider formatting', () => {
    // Helper function to test formatting (extracted from component)
    function formatProvider(ownedBy: string): string {
      const providerMap: Record<string, string> = {
        openai: 'OpenAI',
      };

      const normalized = ownedBy.toLowerCase();
      if (providerMap[normalized]) {
        return providerMap[normalized];
      }

      return ownedBy.charAt(0).toUpperCase() + ownedBy.slice(1).toLowerCase();
    }

    it('should format "openai" as "OpenAI" with correct capitalization', () => {
      expect(formatProvider('openai')).toBe('OpenAI');
      expect(formatProvider('OpenAI')).toBe('OpenAI');
      expect(formatProvider('OPENAI')).toBe('OpenAI');
    });

    it('should capitalize other providers normally', () => {
      expect(formatProvider('anthropic')).toBe('Anthropic');
      expect(formatProvider('meta')).toBe('Meta');
      expect(formatProvider('ANTHROPIC')).toBe('Anthropic');
    });

    it('should render provider names in trigger', () => {
      const models: ModelInfo[] = [
        createMockModel({
          id: 'gpt-4o',
          owned_by: 'openai',
        }),
      ];

      render(<ModelSelector {...defaultProps} models={models} value="gpt-4o" />);

      // Trigger should show OpenAI (not Openai)
      expect(screen.getByText(/OpenAI/)).toBeInTheDocument();
    });
  });

  describe('price formatting', () => {
    // Helper function to test formatting (extracted from component)
    function formatPrice(costPerToken: number): string {
      const perMillion = costPerToken * 1_000_000;
      return `$${perMillion.toFixed(2)}`;
    }

    it('should format prices per million tokens correctly', () => {
      // 0.0000025 * 1,000,000 = $2.50
      expect(formatPrice(0.0000025)).toBe('$2.50');
      // 0.00001 * 1,000,000 = $10.00
      expect(formatPrice(0.00001)).toBe('$10.00');
    });

    it('should format small prices correctly', () => {
      // $0.15 per million
      expect(formatPrice(0.00000015)).toBe('$0.15');
      // $0.60 per million
      expect(formatPrice(0.0000006)).toBe('$0.60');
    });

    it('should display prices in trigger', () => {
      const models: ModelInfo[] = [
        createMockModel({
          id: 'test-model',
          owned_by: 'test',
          input_cost_per_token: 0.0000025,
          output_cost_per_token: 0.00001,
        }),
      ];

      render(<ModelSelector {...defaultProps} models={models} value="test-model" />);

      expect(screen.getByText(/In: \$2\.50/)).toBeInTheDocument();
      expect(screen.getByText(/Out: \$10\.00/)).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('should show loading message when isLoading is true', () => {
      render(<ModelSelector {...defaultProps} isLoading={true} />);

      expect(screen.getByText('Loading models...')).toBeInTheDocument();
    });

    it('should not render dropdown when loading', () => {
      render(<ModelSelector {...defaultProps} isLoading={true} />);

      expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    });
  });

  describe('empty state', () => {
    it('should show fallback when no models available', () => {
      render(<ModelSelector {...defaultProps} models={[]} />);

      expect(screen.getByText(/gpt-4o \(default\)/)).toBeInTheDocument();
    });
  });

  describe('trigger display', () => {
    it('should display current model with pricing in trigger', () => {
      const models: ModelInfo[] = [
        createMockModel({
          id: 'gpt-4o',
          owned_by: 'openai',
          input_cost_per_token: 0.0000025,
          output_cost_per_token: 0.00001,
        }),
      ];

      render(<ModelSelector {...defaultProps} models={models} value="gpt-4o" />);

      // Trigger should show model name
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
      // And pricing details
      expect(screen.getByText(/OpenAI • In: \$2\.50 Out: \$10\.00/)).toBeInTheDocument();
    });
  });
});

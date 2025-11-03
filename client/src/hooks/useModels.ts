/**
 * React hook for fetching available models from the API.
 *
 * Fetches models on mount and provides loading/error states.
 */

import { useState, useEffect } from 'react';
import type { APIClient } from '../lib/api-client';

export interface UseModelsResult {
  models: string[];
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Hook to fetch available models from the API.
 *
 * @param apiClient - Configured API client
 * @returns Models list with loading/error states
 *
 * @example
 * ```typescript
 * const { client } = useAPIClient();
 * const { models, isLoading, error } = useModels(client);
 *
 * if (isLoading) return <div>Loading models...</div>;
 * if (error) return <div>Error: {error}</div>;
 * return <ModelSelector models={models} />;
 * ```
 */
export function useModels(apiClient: APIClient): UseModelsResult {
  const [models, setModels] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchTrigger, setRefetchTrigger] = useState(0);

  useEffect(() => {
    let isMounted = true;
    const abortController = new AbortController();

    const fetchModels = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const fetchedModels = await apiClient.getModels({
          signal: abortController.signal,
        });

        if (isMounted) {
          setModels(fetchedModels);
        }
      } catch (err) {
        // Don't set error if request was aborted (component unmounted)
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }

        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch models');
          // Set empty array on error so we can show fallback
          setModels([]);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchModels();

    return () => {
      isMounted = false;
      abortController.abort();
    };
  }, [apiClient, refetchTrigger]);

  const refetch = () => {
    setRefetchTrigger((prev) => prev + 1);
  };

  return {
    models,
    isLoading,
    error,
    refetch,
  };
}

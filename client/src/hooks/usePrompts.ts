/**
 * React hook for fetching available prompts from the API.
 *
 * Fetches prompts on mount and provides loading/error states.
 */

import { useState, useEffect } from 'react';
import type { APIClient, Prompt } from '../lib/api-client';

export interface UsePromptsResult {
  prompts: Prompt[];
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Hook to fetch available prompts from the API.
 *
 * @param apiClient - Configured API client
 * @returns Prompts list with loading/error states
 *
 * @example
 * ```typescript
 * const { client } = useAPIClient();
 * const { prompts, isLoading, error } = usePrompts(client);
 *
 * if (isLoading) return <div>Loading prompts...</div>;
 * if (error) return <div>Error: {error}</div>;
 * return <CommandPalette prompts={prompts} />;
 * ```
 */
export function usePrompts(apiClient: APIClient): UsePromptsResult {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchTrigger, setRefetchTrigger] = useState(0);

  useEffect(() => {
    let isMounted = true;
    const abortController = new AbortController();

    const fetchPrompts = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const fetchedPrompts = await apiClient.listPrompts({
          signal: abortController.signal,
        });

        if (isMounted) {
          setPrompts(fetchedPrompts);
        }
      } catch (err) {
        // Don't set error if request was aborted (component unmounted)
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }

        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch prompts');
          // Set empty array on error so we can show fallback
          setPrompts([]);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchPrompts();

    return () => {
      isMounted = false;
      abortController.abort();
    };
  }, [apiClient, refetchTrigger]);

  const refetch = () => {
    setRefetchTrigger((prev) => prev + 1);
  };

  return {
    prompts,
    isLoading,
    error,
    refetch,
  };
}

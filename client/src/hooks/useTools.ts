/**
 * React hook for fetching available tools from the API.
 *
 * Fetches tools on mount and provides loading/error states.
 */

import { useState, useEffect } from 'react';
import type { APIClient, Tool } from '../lib/api-client';

export interface UseToolsResult {
  tools: Tool[];
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Hook to fetch available tools from the API.
 *
 * @param apiClient - Configured API client
 * @returns Tools list with loading/error states
 *
 * @example
 * ```typescript
 * const { client } = useAPIClient();
 * const { tools, isLoading, error } = useTools(client);
 *
 * if (isLoading) return <div>Loading tools...</div>;
 * if (error) return <div>Error: {error}</div>;
 * return <CommandPalette tools={tools} />;
 * ```
 */
export function useTools(apiClient: APIClient): UseToolsResult {
  const [tools, setTools] = useState<Tool[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchTrigger, setRefetchTrigger] = useState(0);

  useEffect(() => {
    let isMounted = true;
    const abortController = new AbortController();

    const fetchTools = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const fetchedTools = await apiClient.listTools({
          signal: abortController.signal,
        });

        if (isMounted) {
          setTools(fetchedTools);
        }
      } catch (err) {
        // Don't set error if request was aborted (component unmounted)
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }

        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch tools');
          // Set empty array on error so we can show fallback
          setTools([]);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchTools();

    return () => {
      isMounted = false;
      abortController.abort();
    };
  }, [apiClient, refetchTrigger]);

  const refetch = () => {
    setRefetchTrigger((prev) => prev + 1);
  };

  return {
    tools,
    isLoading,
    error,
    refetch,
  };
}

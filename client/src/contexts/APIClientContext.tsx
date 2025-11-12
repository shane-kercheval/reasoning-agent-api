/**
 * API Client Context for dependency injection.
 *
 * Best practice: Use React Context to provide dependencies (API client)
 * to components instead of creating instances directly. This enables:
 * - Easy mocking in tests
 * - Single source of truth for configuration
 * - Lazy initialization
 */

import { createContext, useContext, ReactNode, useMemo } from 'react';
import { APIClient, createAPIClient } from '../lib/api-client';

interface APIClientContextValue {
  client: APIClient;
}

const APIClientContext = createContext<APIClientContextValue | null>(null);

interface APIClientProviderProps {
  children: ReactNode;
  /** Optional client for testing/mocking */
  client?: APIClient;
}

/**
 * Provider component that makes API client available to all children.
 *
 * @example
 * ```tsx
 * <APIClientProvider>
 *   <App />
 * </APIClientProvider>
 * ```
 */
export function APIClientProvider({
  children,
  client: providedClient,
}: APIClientProviderProps): JSX.Element {
  // Create client only once (or use provided client for testing)
  const client = useMemo(
    () => providedClient || createAPIClient(),
    [providedClient],
  );

  const value = useMemo(() => ({ client }), [client]);

  return (
    <APIClientContext.Provider value={value}>
      {children}
    </APIClientContext.Provider>
  );
}

/**
 * Hook to access the API client from context.
 *
 * @throws Error if used outside APIClientProvider
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { client } = useAPIClient();
 *   // Use client...
 * }
 * ```
 */
export function useAPIClient(): APIClientContextValue {
  const context = useContext(APIClientContext);

  if (!context) {
    throw new Error('useAPIClient must be used within APIClientProvider');
  }

  return context;
}

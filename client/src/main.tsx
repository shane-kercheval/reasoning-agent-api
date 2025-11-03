import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import { ErrorBoundary } from './components/ErrorBoundary.tsx';
import { APIClientProvider } from './contexts/APIClientContext.tsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <APIClientProvider>
        <App />
      </APIClientProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);

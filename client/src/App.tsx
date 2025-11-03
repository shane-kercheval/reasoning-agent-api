import { useState, useEffect } from 'react';

interface Config {
  apiUrl: string;
  apiToken: string;
  nodeEnv: string;
}

function App(): JSX.Element {
  const [config, setConfig] = useState<Config | null>(null);

  useEffect(() => {
    // Get config from Electron preload script
    if (window.electronAPI) {
      setConfig(window.electronAPI.getConfig());
    }
  }, []);

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Reasoning Agent Desktop Client
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Hello World - Milestone 1 Complete! 🎉
        </p>

        {config && (
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-md mx-auto">
            <h2 className="text-lg font-semibold text-gray-700 mb-4">
              Configuration
            </h2>
            <div className="text-left space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">API URL:</span>
                <span className="font-mono text-gray-900">{config.apiUrl}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Environment:</span>
                <span className="font-mono text-gray-900">{config.nodeEnv}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Platform:</span>
                <span className="font-mono text-gray-900">{window.electronAPI?.platform || 'unknown'}</span>
              </div>
            </div>
          </div>
        )}

        <div className="mt-8 text-gray-500 text-sm">
          <p>Built with React + TypeScript + Tailwind CSS + Electron</p>
          <p className="mt-2">Ready for Milestone 2! 🚀</p>
        </div>
      </div>
    </div>
  );
}

export default App;

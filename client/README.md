# Reasoning Agent Client

Electron desktop application for the Reasoning Agent API, built with React, TypeScript, Tailwind CSS, and shadcn/ui.

## Status

**✅ Milestone 1 Complete: Project Scaffolding**

## Features

- **Electron** - Native desktop application
- **React 18** - Modern UI framework with hooks
- **TypeScript (Strict Mode)** - Type-safe development
- **Vite** - Fast build tool with hot module reload
- **Tailwind CSS** - Utility-first styling
- **Jest + React Testing Library** - Comprehensive testing

## Prerequisites

- **Node.js 18+** (client runs natively, not in Docker)
- **Backend services** running via Docker (see root README.md)

## Development

```bash
# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Run tests
npm test

# Run tests in watch mode
npm test:watch

# Type checking
npm run type-check

# Build production app
npm run build
```

## Project Structure

```
client/
├── electron/          # Electron main process and preload script
│   ├── main.ts       # Main process (window creation, security)
│   └── preload.ts    # Preload script (secure IPC bridge)
├── src/              # React application
│   ├── App.tsx       # Root component
│   ├── main.tsx      # React entry point
│   ├── index.css     # Tailwind CSS imports
│   └── electron.d.ts # TypeScript definitions for Electron API
├── tests/            # Test files
│   ├── setup.ts      # Jest configuration
│   └── App.test.tsx  # Component tests
├── public/           # Static assets
├── dist/             # Vite build output (React app)
├── dist-electron/    # Electron build output
├── release/          # Distributable packages (DMG, EXE, AppImage)
└── package.json      # Dependencies and scripts
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Reasoning API Configuration
REASONING_API_URL=http://localhost:8000
REASONING_API_TOKEN=  # Optional - leave empty if backend REQUIRE_AUTH=false

# Development
NODE_ENV=development
```

## Testing

All tests use Jest and React Testing Library:

```bash
npm test              # Run all tests
npm run test:watch    # Watch mode
```

Current test coverage:
- ✅ App renders without crashing
- ✅ Hello World message displays
- ✅ Configuration loads from Electron API

## Security

The application follows Electron security best practices:
- ✅ `nodeIntegration: false` - Prevents Node.js access in renderer
- ✅ `contextIsolation: true` - Isolates preload script context
- ✅ `sandbox: true` - Sandboxes renderer process
- ✅ Controlled IPC via preload script only

## Next Steps (Milestone 2)

- Define OpenAI-compatible API types
- Create HTTP client with SSE streaming
- Implement `useStreamingChat` React hook

## Architecture

The client communicates with the backend API via HTTP:

```
Developer's Machine:
┌─────────────────────────────┐
│  Electron App (native)      │
│  npm run dev                │
└──────────┬──────────────────┘
           │ HTTP
           ▼
    http://localhost:8000
           │
┌──────────┴──────────────────┐
│  Docker Compose Services    │
│  make docker_up             │
│  - reasoning-api            │
│  - litellm                  │
│  - postgres                 │
│  - phoenix                  │
└─────────────────────────────┘
```

## Build Output

Production build creates distributable packages in `release/`:
- macOS: `.dmg` installer
- Windows: `.exe` installer (via `npm run build` on Windows)
- Linux: `.AppImage` (via `npm run build` on Linux)

Current build size: ~92 MB (includes Electron runtime)

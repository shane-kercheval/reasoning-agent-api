import { app, BrowserWindow, session } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Suppress security warnings in development (they're expected for Vite HMR)
// Production builds have strict CSP and full security
if (process.env.VITE_DEV_SERVER_URL) {
  process.env.ELECTRON_DISABLE_SECURITY_WARNINGS = 'true';
}

// Disable hardware acceleration if needed
// app.disableHardwareAcceleration();

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  // Determine preload script path (dev uses .mjs, we need to find it)
  const preloadPath = path.join(__dirname, 'preload.mjs');

  // Set Content Security Policy
  // In development: permissive CSP for Vite HMR (unsafe-inline, unsafe-eval)
  // In production: strict CSP
  const isDev = process.env.VITE_DEV_SERVER_URL !== undefined;
  const csp = isDev
    ? [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https:",
        "connect-src 'self' http://localhost:* ws://localhost:* http://127.0.0.1:* ws://127.0.0.1:*",
      ].join('; ')
    : [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https:",
        "connect-src 'self' http://localhost:* https://*",
      ].join('; ');

  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp],
      },
    });
  });

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: preloadPath,
      // Security settings
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
    // Modern window styling
    titleBarStyle: 'default',
    show: false, // Show after ready-to-show event
  });

  // Show window when ready to prevent flashing
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  // Load the app
  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    // Open DevTools in development
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    // On macOS, re-create window when dock icon is clicked and no windows are open
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Security: Prevent navigation to external URLs
app.on('web-contents-created', (_, contents) => {
  contents.on('will-navigate', (event, navigationUrl) => {
    const parsedUrl = new URL(navigationUrl);

    // Allow navigation within the app only
    if (parsedUrl.origin !== 'file://') {
      event.preventDefault();
    }
  });
});

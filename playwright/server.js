#!/usr/bin/env node
/**
 * Playwright Network Server
 * Exposes a WebSocket API for remote browser control.
 * Agents on other machines can connect via ws://<IP>:<PORT>/
 *
 * Usage:
 *   node server.js [--port 3001] [--host 0.0.0.0]
 */

const { spawn } = require('child_process');
const path = require('path');

// Parse args
let PORT = parseInt(process.env.PLAYWRIGHT_PORT || '3001', 10);
let HOST = process.env.PLAYWRIGHT_HOST || '0.0.0.0';
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--port' && process.argv[i + 1]) {
    PORT = parseInt(process.argv[++i], 10);
  } else if (process.argv[i] === '--host' && process.argv[i + 1]) {
    HOST = process.argv[++i];
  }
}

const LOG_DIR = path.join(__dirname, 'logs');
require('fs').mkdirSync(LOG_DIR, { recursive: true });

// Resolve playwright CLI
const pwBin = path.join(__dirname, 'node_modules', '.bin', 'playwright');

console.log(`[playwright-server] Starting on ws://${HOST}:${PORT}/`);
console.log('[playwright-server] Browser: Chromium (headless)');
console.log('[playwright-server] Log: ' + LOG_DIR + '/server.log');
console.log('[playwright-server] PID: ' + process.pid);
console.log('');
console.log('[playwright-server] Remote agents can connect with:');
console.log(`[playwright-server]   const browser = await playwright.chromium.connect('ws://${HOST}:${PORT}/');`);
console.log('');

const server = spawn(pwBin, ['run-server', '--host', HOST, '--port', String(PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
  env: {
    ...process.env,
    PLAYWRIGHT_BROWSERS_PATH: process.env.PLAYWRIGHT_BROWSERS_PATH || '0',
  },
});

server.stdout.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.log(`[server] ${line}`);
});

server.stderr.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.error(`[server] ${line}`);
});

server.on('close', (code) => {
  console.log(`[playwright-server] Process exited with code ${code}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[playwright-server] Shutting down...');
  server.kill('SIGTERM');
  setTimeout(() => process.exit(0), 2000);
});

process.on('SIGTERM', () => {
  console.log('\n[playwright-server] Shutting down...');
  server.kill('SIGTERM');
  setTimeout(() => process.exit(0), 2000);
});

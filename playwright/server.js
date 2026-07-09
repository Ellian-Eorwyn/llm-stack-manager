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
const http = require('http');
const net = require('net');
const path = require('path');

// Parse args
let PORT = parseInt(process.env.PLAYWRIGHT_PORT || '3001', 10);
let HOST = process.env.PLAYWRIGHT_HOST || '0.0.0.0';
let URL_PATH = process.env.PLAYWRIGHT_URL_PATH || '/playwright';
let UPSTREAM_PORT = parseInt(process.env.PLAYWRIGHT_UPSTREAM_PORT || String(PORT + 10000), 10);
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--port' && process.argv[i + 1]) {
    PORT = parseInt(process.argv[++i], 10);
  } else if (process.argv[i] === '--host' && process.argv[i + 1]) {
    HOST = process.argv[++i];
  } else if (process.argv[i] === '--path' && process.argv[i + 1]) {
    URL_PATH = process.argv[++i];
  } else if (process.argv[i] === '--upstream-port' && process.argv[i + 1]) {
    UPSTREAM_PORT = parseInt(process.argv[++i], 10);
  }
}
if (!URL_PATH.startsWith('/')) URL_PATH = `/${URL_PATH}`;
URL_PATH = URL_PATH.replace(/\/+$/, '') || '/playwright';

const LOG_DIR = path.join(__dirname, 'logs');
require('fs').mkdirSync(LOG_DIR, { recursive: true });

// Resolve playwright CLI
const pwBin = path.join(__dirname, 'node_modules', '.bin', 'playwright');

console.log(`[playwright-server] Starting on ws://${HOST}:${PORT}/`);
console.log(`[playwright-server] Path alias: ${URL_PATH}/ -> /`);
console.log(`[playwright-server] Internal upstream: ws://127.0.0.1:${UPSTREAM_PORT}/`);
console.log('[playwright-server] Browser: Chromium (headless)');
console.log('[playwright-server] Log: ' + LOG_DIR + '/server.log');
console.log('[playwright-server] PID: ' + process.pid);
console.log('');
console.log('[playwright-server] Remote agents can connect with:');
console.log(`[playwright-server]   const browser = await playwright.chromium.connect('ws://${HOST}:${PORT}/');`);
console.log(`[playwright-server]   const browser = await playwright.chromium.connect('ws://${HOST}${URL_PATH}/');`);
console.log('');

const upstream = spawn(pwBin, ['run-server', '--host', '127.0.0.1', '--port', String(UPSTREAM_PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
  env: {
    ...process.env,
    PLAYWRIGHT_BROWSERS_PATH: process.env.PLAYWRIGHT_BROWSERS_PATH || '0',
  },
});

upstream.stdout.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.log(`[upstream] ${line}`);
});

upstream.stderr.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.error(`[upstream] ${line}`);
});

upstream.on('close', (code) => {
  console.log(`[playwright-server] Upstream process exited with code ${code}`);
  process.exit(code || 0);
});

function rewritePath(originalPath) {
  if (originalPath === URL_PATH) return '/';
  if (originalPath.startsWith(`${URL_PATH}/`)) return originalPath.slice(URL_PATH.length) || '/';
  return originalPath || '/';
}

const proxy = http.createServer((req, res) => {
  const upstreamPath = rewritePath(req.url || '/');
  const upstreamReq = http.request({
    host: '127.0.0.1',
    port: UPSTREAM_PORT,
    method: req.method,
    path: upstreamPath,
    headers: {
      ...req.headers,
      host: `127.0.0.1:${UPSTREAM_PORT}`,
    },
  }, (upstreamRes) => {
    res.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
    upstreamRes.pipe(res);
  });
  upstreamReq.on('error', (err) => {
    res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(`Playwright upstream unavailable: ${err.message}\n`);
  });
  req.pipe(upstreamReq);
});

proxy.on('upgrade', (req, socket, head) => {
  const upstreamSocket = net.connect(UPSTREAM_PORT, '127.0.0.1');
  upstreamSocket.on('connect', () => {
    const upstreamPath = rewritePath(req.url || '/');
    const headers = [
      `${req.method} ${upstreamPath} HTTP/${req.httpVersion}`,
      ...Object.entries(req.headers)
        .filter(([name]) => name.toLowerCase() !== 'host')
        .map(([name, value]) => `${name}: ${Array.isArray(value) ? value.join(', ') : value}`),
      `Host: 127.0.0.1:${UPSTREAM_PORT}`,
      '',
      '',
    ].join('\r\n');
    upstreamSocket.write(headers);
    if (head.length) upstreamSocket.write(head);
    socket.pipe(upstreamSocket);
    upstreamSocket.pipe(socket);
  });
  upstreamSocket.on('error', (err) => {
    socket.write('HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n');
    socket.destroy(err);
  });
});

proxy.listen(PORT, HOST, () => {
  console.log(`[playwright-server] Listening on ws://${HOST}:${PORT}/`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[playwright-server] Shutting down...');
  proxy.close();
  upstream.kill('SIGTERM');
  setTimeout(() => process.exit(0), 2000);
});

process.on('SIGTERM', () => {
  console.log('\n[playwright-server] Shutting down...');
  proxy.close();
  upstream.kill('SIGTERM');
  setTimeout(() => process.exit(0), 2000);
});

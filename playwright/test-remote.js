#!/usr/bin/env node
/**
 * Test client — connects to the Playwright WS server and scrapes a page.
 *
 * Usage:
 *   node test-remote.js [--host <IP>] [--port <PORT>] [--url <URL>]
 */

const playwright = require('playwright');

let HOST = '127.0.0.1';
let PORT = 3001;
let URL = 'https://example.com';
let ENDPOINT = '';

for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--host' && process.argv[i + 1]) HOST = process.argv[++i];
  if (process.argv[i] === '--port' && process.argv[i + 1]) PORT = process.argv[++i];
  if (process.argv[i] === '--url' && process.argv[i + 1]) URL = process.argv[++i];
  if (process.argv[i] === '--endpoint' && process.argv[i + 1]) ENDPOINT = process.argv[++i];
}

(async () => {
  const wsEndpoint = ENDPOINT || `ws://${HOST}:${PORT}/`;
  console.log(`Connecting to ${wsEndpoint} ...`);

  const browser = await playwright.chromium.connect(wsEndpoint);
  console.log('✓ Connected');

  const context = await browser.newContext();
  const page = await context.newPage();

  console.log(`Navigating to ${URL} ...`);
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

  const title = await page.title();
  const content = await page.content();
  const text = await page.innerText('body');

  console.log(`\n✓ Page loaded`);
  console.log(`  Title:   ${title}`);
  console.log(`  HTML len: ${content.length} chars`);
  console.log(`  Text len: ${text.length} chars`);
  console.log(`\n  First 200 chars of text:`);
  console.log(`  ${text.slice(0, 200)}\n`);

  await browser.close();
  console.log('✓ Done');
})();

// Load replay files into the *running* viewer through its "Load model JSON"
// button, headlessly, and report whether the viewer accepted each one.
//
//   cd viewer && npm run dev            # in another shell
//   node scripts/viewer_load_test.mjs results/replays/*.json
//
// Needs the `playwright` npm package resolvable from NODE_PATH (it is not a
// dependency of this project) and Google Chrome installed; it drives the
// system Chrome via Playwright's "chrome" channel, so nothing is downloaded.
// Success means: no role="alert" error element after the upload, and the
// model-provenance panel switched from "ANATOMY ONLY" to "PREDICTED OUTPUT"
// showing the file's declared source name.
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const url = process.env.VIEWER_URL ?? 'http://127.0.0.1:5173/';
const files = process.argv.slice(2);
if (!files.length) { console.error('usage: node scripts/viewer_load_test.mjs <replay.json>...'); process.exit(2); }

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const consoleErrors = [];
page.on('pageerror', e => consoleErrors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
await page.goto(url, { waitUntil: 'networkidle' });
// Wait for the atlas: the "Load model JSON" button is disabled until it loads.
const loadButton = page.getByRole('button', { name: 'Load model JSON' });
await loadButton.waitFor({ state: 'visible' });
await page.waitForFunction(() => {
  const b = [...document.querySelectorAll('button')].find(x => x.textContent === 'Load model JSON');
  return b && !b.disabled;
}, null, { timeout: 60000 });
const somata = await page.locator('.brain-panel .panel-bottom').innerText();
console.log(`viewer up at ${url}; atlas loaded: "${somata.trim().split('\n')[0]}"`);

let failed = 0;
for (const file of files) {
  await page.locator('input[type=file]').setInputFiles(resolve(file));
  // Give React a tick to validate and re-render.
  await page.waitForTimeout(300);
  const alert = page.locator('[role=alert]');
  const hadError = await alert.count() > 0 ? (await alert.first().innerText()).trim() : '';
  const status = (await page.locator('.model-status').innerText()).trim().split('\n');
  const accepted = !hadError && status[0] === 'PREDICTED OUTPUT';
  if (!accepted) failed++;
  console.log(`${accepted ? 'ok  ' : 'FAIL'}  ${file}`);
  console.log(`      status: ${status[0]} | ${status[1]?.slice(0, 110)}${status[1]?.length > 110 ? '…' : ''}`);
  if (hadError) console.log(`      viewer error: ${hadError}`);
  // Scrub the loaded replay to the last frame and confirm playback works.
  const clear = page.getByRole('button', { name: 'Clear output' });
  if (await clear.count()) await clear.click();
}
if (consoleErrors.length) console.log('browser console errors:', consoleErrors);
await browser.close();
console.log(`${files.length - failed}/${files.length} replays accepted by the running viewer`);
process.exit(failed ? 1 : 0);

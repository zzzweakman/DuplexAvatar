// Browser regression: faster idle decoding must not overtake earlier postroll.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, '..', 'runtime', `playback-order-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(output, {recursive: true});

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required'],
  });
  const context = await browser.newContext({ignoreHTTPSErrors: true});
  const page = await context.newPage(), errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.route('**/health', route => route.fulfill({json: {status: 'ready', audio_ready: true, busy: false,
    brain: {status: 'ready'}, avatar: {status: 'ready'}, asr: {status: 'ok'}}}));
  await page.route('**/static/app.js', async route => {
    const response = await route.fetch();
    let source = await response.text();
    source = source.replace('return decodeVideoBlock(message);',
      'if (message.test_hold) await new Promise(resolve => { window.__releaseDecode = resolve; }); return decodeVideoBlock(message);');
    const needle = 'lastDisplayed = {sequence: latest.sequence, frame_index: latest.frame_index};';
    source = source.replace(needle, `window.__draws.push({at: performance.now(), sequence: latest.sequence, index: latest.frame_index}); ${needle}`);
    await route.fulfill({response, body: source});
  });
  await context.addInitScript(() => {
    window.__draws = [];
    window.WebSocket = class extends EventTarget {
      static OPEN = 1; static CLOSED = 3;
      constructor() {
        super(); this.readyState = 0; this.bufferedAmount = 0; window.__control = this;
        setTimeout(() => { this.readyState = 1; this.onopen?.(); }, 0);
      }
      send(raw) {
        const message = JSON.parse(raw);
        if (message.type === 'session.start') setTimeout(() => this.emit({type: 'session.ready', epoch: 0}), 0);
        if (message.type === 'input') this.emit({type: 'input.received', client_input_id: message.client_input_id, pending_inputs: 0});
        if (message.type === 'session.stop') this.close();
      }
      emit(message) { this.onmessage?.({data: JSON.stringify(message)}); }
      close() { this.readyState = 3; this.onclose?.(); this.dispatchEvent(new Event('close')); }
    };
    window.__block = (sequence, type, test_hold = false) => {
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 64;
      const drawing = canvas.getContext('2d');
      const frames = Array.from({length: 24}, (_, i) => {
        drawing.fillStyle = `rgb(${sequence * 45},${i * 7},80)`; drawing.fillRect(0, 0, 64, 64);
        return canvas.toDataURL('image/jpeg').split(',')[1];
      });
      return {type, sequence, epoch: 0, frames, fps: 25, width: 64, height: 64, audio: '',
        sample_rate: 24000, duration_samples: 23040, response_id: 'reply', rtf: .1, test_hold};
    };
  });
  try {
    await page.goto('https://127.0.0.1:8443', {waitUntil: 'networkidle'});
    await page.locator('#demo').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('进行中'));
    await page.evaluate(() => window.__control.emit(window.__block(1, 'idle')));
    await page.waitForFunction(() => window.__draws.length >= 5);
    await page.evaluate(() => {
      window.__control.emit(window.__block(2, 'media', true));
      window.__control.emit(window.__block(3, 'idle'));
    });
    await page.waitForFunction(() => !!window.__releaseDecode);
    await page.waitForTimeout(1400);
    assert.equal(await page.evaluate(() => window.__draws.some(d => d.sequence === 3)), false,
      'Idle overtook the speech/postroll block waiting for decoding');
    await page.evaluate(() => window.__releaseDecode());
    await page.waitForFunction(() => window.__draws.some(d => d.sequence === 3));
    const draws = await page.evaluate(() => window.__draws);
    assert.ok(draws.findIndex(d => d.sequence === 2) < draws.findIndex(d => d.sequence === 3));
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    assert.deepEqual(errors, []);
    const result = {passed: true, delayed_postroll_decode_ms: 1400, idle_did_not_overtake: true,
      postroll_then_idle: true, page_errors: errors, output};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({result, draws}, null, 2), {flag: 'wx'});
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });

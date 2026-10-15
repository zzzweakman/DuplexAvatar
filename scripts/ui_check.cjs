// Final static UI check and a deterministic regression for cancelled demos.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const output = path.resolve(__dirname, '..', 'runtime', `ui-final-${new Date().toISOString().replace(/[:.]/g, '-')}`);
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required',
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  try {
    const context = await browser.newContext({ignoreHTTPSErrors: true, viewport: {width: 1440, height: 1000}, permissions: ['microphone', 'camera']});
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto('https://127.0.0.1:8443', {waitUntil: 'networkidle'});
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.evaluate(() => document.fonts.check('15px "Noto Sans SC"', '正在对话')), true);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#audioInput').check();
    await page.locator('#useAvatar').uncheck();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    await page.locator('#useAvatar').check();
    await page.locator('#useCamera').check();

    // Use a protocol stub: this regression needs no model inference or GPU.
    await page.evaluate(() => {
      window.__connections = [];
      window.WebSocket = class {
        static OPEN = 1;
        constructor() {
          this.readyState = 0; this.bufferedAmount = 0; this.inputs = 0; window.__connections.push(this);
          setTimeout(() => { this.readyState = 1; this.onopen?.(); }, 0);
        }
        send(raw) {
          const data = JSON.parse(raw);
          if (data.type === 'session.start') {
            this.options = data;
            setTimeout(() => this.onmessage?.({data: JSON.stringify({type: 'session.ready', epoch: 0})}), 0);
          }
          if (data.type === 'input') {
            this.inputs++;
            const pending = window.__forcePending ?? 0;
            setTimeout(() => this.onmessage?.({data: JSON.stringify({type: 'input.received', client_input_id: data.client_input_id,
              pending_inputs: pending, received_inputs: this.inputs, processed_inputs: Math.max(0, this.inputs - pending)})}), 0);
          }
        }
        close() { this.readyState = 3; this.onclose?.(); }
      };
    });
    await page.waitForFunction(() => !document.getElementById('demo').disabled);
    await page.locator('#demo').click();
    await page.waitForFunction(() => window.__connections[0]?.inputs === 1);
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    let heldAudioRequest;
    await page.route('**/demo/audio.wav', route => { heldAudioRequest = route; });
    await page.locator('#demo').click();
    await page.waitForFunction(() => window.__connections.length === 2);
    await page.waitForTimeout(1200);
    const inputs = await page.evaluate(() => window.__connections.map(connection => connection.inputs));
    assert.deepEqual(inputs, [1, 0], 'Cancelled demo leaked input into the new session');
    assert.ok(heldAudioRequest, 'New demo did not request its own audio');
    await heldAudioRequest.continue();
    await page.waitForFunction(() => window.__connections[1]?.inputs === 1);
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    await page.evaluate(() => {
      window.__deviceRequests = [];
      const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async options => {
        window.__deviceRequests.push(options);
        if (options.video) throw new DOMException('Camera unavailable', 'NotFoundError');
        return original(options);
      };
    });
    await page.locator('#start').click();
    await page.waitForFunction(() => parseInt(document.getElementById('inputCount').textContent) >= 1);
    assert.equal(await page.locator('#audioInput').isChecked(), true);
    assert.equal(await page.locator('#cameraPanel').isVisible(), false);
    assert.equal(await page.evaluate(() => window.__connections.at(-1).options.input_mode), 'audio');
    assert.equal(await page.evaluate(() => window.__deviceRequests.at(-1).video), false);
    // A slow model must not terminate capture at six seconds; high water pauses and then resumes it.
    await page.evaluate(() => { window.__forcePending = 8; });
    await page.waitForFunction(() => document.getElementById('inputDelivery').textContent.includes('8 秒'));
    assert.equal(await page.locator('#stop').isEnabled(), true);
    assert.equal(await page.locator('#notice').evaluate(node => node.classList.contains('error')), false);
    await page.evaluate(() => { window.__forcePending = 24; });
    await page.waitForFunction(() => document.getElementById('micState').textContent.includes('暂缓'));
    const pausedCount = await page.evaluate(() => window.__connections.at(-1).inputs);
    await page.waitForTimeout(1500);
    assert.equal(await page.evaluate(() => window.__connections.at(-1).inputs), pausedCount);
    await page.evaluate(() => {
      window.__forcePending = 0;
      window.__connections.at(-1).onmessage({data: JSON.stringify({type: 'input.processed', pending_inputs: 0, received_inputs: 24})});
    });
    await page.waitForFunction(count => window.__connections.at(-1).inputs > count, pausedCount);
    await page.locator('#stop').click();
    assert.deepEqual(errors, []);
    const result = {passed: true, chinese_font: true, desktop_and_mobile_no_overflow: true,
      cancelled_demo_isolated: true, camera_failure_uses_audio_protocol: true,
      eight_second_backlog_keeps_session: true, capture_high_water_pause_and_resume: true,
      session_input_counts: inputs, page_errors: errors, output};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(result, null, 2), {flag: 'wx'});
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

// Real MiniCPM + A2V: verify long replies do not skip speculative idle motion.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'runtime', `continuity-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(output, {recursive: true});

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required'],
  });
  const context = await browser.newContext({ignoreHTTPSErrors: true, viewport: {width: 1440, height: 1000}});
  await context.addInitScript(() => {
    window.__motion = [];
    const Socket = window.WebSocket;
    window.WebSocket = class extends Socket {
      send(raw) {
        const m = JSON.parse(raw);
        if (m.type === 'session.start') m.system_prompt = '你是中文助手。用户提问图片颜色后，请先回答颜色，再连续介绍这种颜色在生活中十种常见的事物，每种具体介绍一句，至少两百字。不要提问，直接完整讲完。';
        return super.send(JSON.stringify(m));
      }
    };
  });
  const page = await context.newPage(); page.setDefaultTimeout(45000);
  await page.route('**/static/app.js', async route => {
    const response = await route.fetch();
    const source = await response.text();
    const needle = 'lastDisplayed = {sequence: latest.sequence, frame_index: latest.frame_index};';
    assert.ok(source.includes(needle));
    await route.fulfill({response, body: source.replace(needle,
      `window.__motion.push({at: performance.now(), epoch, sequence: latest.sequence, index: latest.frame_index, idle: !!latest.idle, fallback: !!latest.fallback, bridge: !!latest.bridge});\n${needle}`)});
  });
  const events = [], errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('websocket', ws => ws.on('framereceived', event => {
    try {
      const data = JSON.parse(event.payload);
      if (['media', 'idle', 'response.end', 'session.ready', 'text', 'error'].includes(data.type)) events.push({...data, at: Date.now()});
    } catch {}
  }));
  try {
    await page.goto('https://127.0.0.1:8443', {waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.getElementById('demo').disabled);
    await page.locator('#demo').click();
    for (let i = 0; i < 33; i++) {
      await page.waitForTimeout(1000);
      if (i % 10 === 0) console.log(JSON.stringify({second: i, speech_blocks: events.filter(e => e.type === 'media').length}));
    }
    const draws = await page.evaluate(() => window.__motion);
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    const speech = events.filter(e => e.type === 'media'), between = [];
    for (let i = 1; i < speech.length; i++) {
      const previous = speech[i - 1], next = speech[i];
      if (previous.response_id === next.response_id) {
        const idle = events.filter(e => e.type === 'idle' && e.sequence > previous.sequence && e.sequence < next.sequence);
        if (idle.length) between.push({previous: previous.sequence, next: next.sequence, idle: idle.map(e => e.sequence)});
      }
    }
    const backward = draws.slice(1).filter((d, i) => d.epoch === draws[i].epoch && !d.bridge && !draws[i].bridge &&
      (d.sequence < draws[i].sequence || d.sequence === draws[i].sequence && d.index < draws[i].index));
    const summary = {output, session: events.find(e => e.type === 'session.ready')?.session_id,
      speech_blocks: speech.length, idle_insertions_inside_reply: between.length,
      backwards_presentations: backward.length, fallback_frames: draws.filter(d => d.fallback).length,
      page_errors: errors, between};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({summary, draws, events}, null, 2), {flag: 'wx'});
    console.log(JSON.stringify(summary, null, 2));
    assert.deepEqual(errors, []);
    assert.ok(speech.length >= 3, 'A reply spanning multiple A2V blocks was not generated');
    if (process.argv.includes('--verify')) {
      assert.equal(between.length, 0, 'Idle rendering advanced motion inside a continuous reply');
      assert.equal(backward.length, 0, 'Frames moved backwards under ordinary delivery');
    }
  } finally {
    if (await page.locator('#stop').isEnabled().catch(() => false)) {
      await page.locator('#stop').click();
      await page.waitForFunction(() => !document.getElementById('start').disabled).catch(() => {});
    }
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

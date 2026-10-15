// Measure actual browser draws and visible motion while microphone speech is active.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'runtime', `listening-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(output, {recursive: true});
const fixture = path.join(output, 'microphone.wav');
const wav = spawnSync('ffmpeg', ['-nostdin', '-v', 'error', '-n', '-stream_loop', '5', '-i',
  path.join(root, 'runtime/fixtures/question.wav'), '-af', 'adelay=10000:all=1,apad=pad_dur=12',
  '-t', '42', '-ar', '16000', '-ac', '1', fixture]);
assert.equal(wav.status, 0, wav.stderr?.toString());

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required',
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${fixture}%noloop`],
  });
  const context = await browser.newContext({ignoreHTTPSErrors: true, viewport: {width: 1440, height: 1000}, permissions: ['microphone']});
  await context.addInitScript(() => {
    window.__draws = []; window.__samples = []; window.__contexts = []; window.__ticks = []; window.__record = false;
    const tick = now => { if (window.__record) window.__ticks.push(now); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    const draw = CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage = function(...args) {
      if (window.__record && this.canvas.id === 'avatar') window.__draws.push({at: performance.now(), state: this.canvas.dataset.motion,
        source: this.canvas.dataset.videoSource});
      return draw.apply(this, args);
    };
    const Context = window.AudioContext;
    window.AudioContext = class extends Context {
      constructor(options) { super(options); window.__contexts.push(this); }
    };
    addEventListener('DOMContentLoaded', () => {
      const small = document.createElement('canvas'); small.width = small.height = 64;
      const drawing = small.getContext('2d', {willReadFrequently: true});
      let previous;
      window.__timer = setInterval(() => {
        if (!window.__record) return;
        const avatar = document.getElementById('avatar');
        drawing.drawImage(avatar, 0, 0, 64, 64);
        const pixels = drawing.getImageData(0, 0, 64, 64).data;
        let delta = 0;
        if (previous) for (let i = 0; i < pixels.length; i++) if (i % 4 !== 3) delta += Math.abs(pixels[i] - previous[i]);
        previous = pixels;
        window.__samples.push({at: performance.now(), state: avatar.dataset.motion,
          frames: parseInt(document.getElementById('frameCount').textContent),
          db: Number(document.getElementById('inputMeter').getAttribute('aria-valuenow')),
          label: document.getElementById('playState').textContent, mae: delta / (64 * 64 * 3),
          contexts: window.__contexts.map(c => ({state: c.state, time: c.currentTime}))});
      }, 50);
    });
  });
  if (process.argv.includes('--stall-media')) await context.addInitScript(() => {
    const Socket = window.WebSocket;
    window.WebSocket = class extends Socket {
      get onmessage() { return this.handler; }
      set onmessage(handler) {
        this.handler = handler;
        super.onmessage = event => {
          const elapsed = performance.now() - (window.__testStarted || Infinity);
          if (this.url.endsWith('/media') && elapsed >= 15000 && elapsed < 19000) {
            (this.held ||= []).push(event);
            if (!this.release) this.release = setTimeout(() => {
              this.release = null;
              for (const pending of this.held.splice(0)) this.handler(pending);
            }, 19000 - elapsed);
          } else handler(event);
        };
      }
    };
  });
  const page = await context.newPage(); page.setDefaultTimeout(30000);
  const events = [], errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('websocket', ws => {
    for (const [eventName, direction] of [['framereceived', 'received'], ['framesent', 'sent']]) ws.on(eventName, event => {
      try {
        const m = JSON.parse(event.payload);
        if (['input', 'input.received', 'playback'].includes(m.type)) return;
        const {audio, frames, video, ...data} = m;
        events.push({...data, direction, at: Date.now(), frame_count: video?.frame_count ?? frames?.length});
      } catch {}
    });
  });
  try {
    await page.goto('https://127.0.0.1:8443', {waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    await page.locator('#audioInput').check();
    await page.locator('#start').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('进行中'));
    await page.evaluate(() => { window.__record = true; window.__testStarted = performance.now(); });
    if (process.argv.includes('--stall-media')) await page.evaluate(() => {
      setTimeout(async () => {
        const playback = window.__contexts[0]; await playback.suspend();
        window.__clockPause = performance.now();
        setTimeout(() => playback.resume(), 1000);
      }, 5000);
    });
    for (let i = 0; i < 42; i++) {
      await page.waitForTimeout(1000);
      if (process.argv.includes('--capture')) {
        const image = await page.locator('#avatar').evaluate(canvas => canvas.toDataURL('image/jpeg', .85));
        fs.writeFileSync(path.join(output, `avatar-${String(i).padStart(3, '0')}.jpg`), Buffer.from(image.split(',')[1], 'base64'), {flag: 'wx'});
      }
      if (i % 10 === 0) console.log(JSON.stringify({second: i, last: await page.evaluate(() => window.__samples.at(-1))}));
    }
    const measured = await page.evaluate(() => {
      window.__record = false;
      return {draws: window.__draws, samples: window.__samples, ticks: window.__ticks};
    });
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    const spoken = measured.samples.filter(s => s.db > -35);
    const gaps = measured.draws.slice(1).map((draw, i) => draw.at - measured.draws[i].at);
    const tickGaps = measured.ticks.slice(1).map((tick, i) => tick - measured.ticks[i]);
    const responsiveGaps = measured.draws.slice(1).map((draw, i) => {
      const previous = measured.draws[i].at;
      // Count animation opportunities without a canvas update. A paused browser
      // cannot draw; report that wall-clock gap separately instead of attributing
      // OS scheduling or instrumentation stalls to the media pipeline.
      const ticks = measured.ticks.filter(t => t > previous + 1 && t < draw.at - 1);
      return ticks.length * 1000 / 60;
    });
    const summary = {output, errors, draws: measured.draws.length,
      max_draw_gap_ms: Math.max(0, ...gaps), voiced_samples: spoken.length,
      max_browser_tick_gap_ms: Math.max(0, ...tickGaps),
      max_undrawn_animation_ms: Math.max(0, ...responsiveGaps),
      voiced_moving_samples: spoken.filter(s => s.mae > .05).length,
      voiced_idle_samples: spoken.filter(s => s.state === 'idle').length,
      fallback_draws: measured.draws.filter(d => d.source === 'buffered-idle').length,
      injected_media_delay_ms: process.argv.includes('--stall-media') ? 4000 : 0,
      injected_output_clock_pause_ms: process.argv.includes('--stall-media') ? 1000 : 0,
      received_idle_blocks: events.filter(e => e.direction === 'received' && e.type === 'idle').length};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({summary, ...measured, events}, null, 2), {flag: 'wx'});
    console.log(JSON.stringify(summary, null, 2));
    assert.deepEqual(errors, []);
    if (process.argv.includes('--verify')) {
      assert.ok(summary.max_undrawn_animation_ms < 250, `Avatar missed ${summary.max_undrawn_animation_ms} ms of animation opportunities`);
      assert.ok(spoken.length >= 100, 'Insufficient microphone speech in the test');
      assert.ok(summary.voiced_moving_samples / spoken.length > .8, 'Listening motion stopped during user speech');
      if (process.argv.includes('--stall-media')) assert.ok(summary.fallback_draws > 10, 'Silent recovery was not exercised');
    }
  } finally {
    if (await page.locator('#stop').isEnabled().catch(() => false)) {
      await page.locator('#stop').click();
      await page.waitForFunction(() => !document.getElementById('start').disabled).catch(() => {});
    }
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

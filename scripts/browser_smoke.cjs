// Optional Playwright tooling; the app itself has no frontend build step.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'runtime', `browser-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(output, {recursive: true});

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true,
    env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required',
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${path.join(root, 'runtime/fixtures/question.wav')}%noloop`],
  });
  const context = await browser.newContext({ignoreHTTPSErrors: true, viewport: {width: 1440, height: 1000}, permissions: ['camera', 'microphone']});
  await context.addInitScript(() => {
    window.__playback = {starts: 0, stops: 0, contexts: []};
    window.__devices = [];
    window.__postrollDrawn = false; window.__maxInputDb = -60;
    window.__motionEvents = []; window.__motionDraws = []; window.__recordMotion = false;
    const draw = CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage = function(...args) {
      if (this.canvas.id === 'avatar' && window.__recordMotion) window.__motionDraws.push(Date.now());
      return draw.apply(this, args);
    };
    addEventListener('DOMContentLoaded', () => {
      new MutationObserver(() => {
        if (document.getElementById('avatar')?.dataset.motion === 'postroll') window.__postrollDrawn = true;
        if (window.__recordMotion) window.__motionEvents.push({state: document.getElementById('avatar')?.dataset.motion, at: Date.now()});
        window.__maxInputDb = Math.max(window.__maxInputDb, Number(document.getElementById('inputMeter')?.getAttribute('aria-valuenow') || -60));
      }).observe(document.body, {attributes: true, subtree: true, attributeFilter: ['data-motion', 'aria-valuenow']});
    });
    const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async constraints => {
      const stream = await getUserMedia(constraints);
      window.__devices.push({constraints, videoTracks: stream.getVideoTracks().length, audioTracks: stream.getAudioTracks().length});
      return stream;
    };
    const Context = window.AudioContext;
    window.AudioContext = class extends Context {
      constructor(options) { super(options); window.__playback.contexts.push(this); }
      createBufferSource() {
        const source = super.createBufferSource(), start = source.start.bind(source), stop = source.stop.bind(source);
        source.start = (...args) => { window.__playback.starts++; return start(...args); };
        source.stop = (...args) => { window.__playback.stops++; window.__playback.stoppedAt = Date.now(); return stop(...args); };
        return source;
      }
    };
  });
  const page = await context.newPage();
  page.setDefaultTimeout(90000);
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  const events = [];
  const sentInterrupts = [];
  page.on('websocket', socket => {
    socket.on('framesent', event => {
      try { const data = JSON.parse(event.payload);
        if (data.type === 'interrupt') sentInterrupts.push({...data, sentAt: Date.now()});
      } catch {}
    });
    socket.on('framereceived', event => {
    try { const data = JSON.parse(event.payload); events.push({type: data.type, epoch: data.epoch, channel: socket.url().endsWith('/media') ? 'media' : 'control',
      frames: data.video?.frame_count ?? data.frames?.length, video_codec: data.video?.codec, output_mode: data.output_mode, postroll_samples: data.postroll_samples,
      motion_resume: data.motion_resume, transition: data.transition, timing: data.timing,
      pending_inputs: data.pending_inputs, generated_at_ms: data.generated_at_ms,
      start_ms: data.start_ms, receivedAt: Date.now()}); } catch {}
    });
  });
  try {
    await page.goto('https://127.0.0.1:8443', {waitUntil: 'networkidle'});
    await page.locator('#demo').waitFor({state: 'visible'});
    await page.waitForFunction(() => !document.getElementById('demo').disabled);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.locator('#demo').click();
    await page.waitForFunction(() => document.getElementById('blockClock').textContent.includes('960'));
    await page.waitForFunction(() => document.getElementById('avatar').dataset.motion === 'idle');
    await page.waitForFunction(() => parseInt(document.getElementById('frameCount').textContent) >= 25, {timeout: 90000});
    await page.waitForFunction(() => document.getElementById('transcript').textContent.includes('红'), {timeout: 90000});
    assert.ok(events.some(e => e.type === 'media.ready' && e.channel === 'media'));
    assert.ok(events.some(e => e.type === 'input.processed' && e.channel === 'control'));
    assert.ok(events.filter(e => ['media', 'idle', 'response.end'].includes(e.type)).every(e => e.channel === 'media'));
    assert.ok(events.some(e => e.video_codec?.startsWith('avc1.')));
    const playback = await page.evaluate(() => ({starts: window.__playback.starts, states: window.__playback.contexts.map(c => c.state)}));
    assert.ok(playback.starts > 0 && playback.states.includes('running'), JSON.stringify(playback));
    await page.waitForFunction(() => window.__postrollDrawn);
    await page.locator('.turn[data-role="user"] p').first().waitFor();
    assert.match(await page.locator('.turn[data-role="user"] p').first().textContent(), /颜色/);
    assert.match(await page.locator('.turn[data-role="user"] time').first().textContent(), /\d{2}:\d{2}:\d{2}.*\d{2}:\d{2}\.\d/);
    assert.match(await page.locator('.turn[data-role="assistant"] .turn-meta time').first().textContent(), /生成 \d{2}:\d{2}:\d{2}/);
    assert.match(await page.locator('.turn-played').first().textContent(), /播放 \d{2}:\d{2}:\d{2}/);
    assert.ok(await page.evaluate(() => window.__maxInputDb > -44));
    assert.match(await page.locator('#inputDelivery').textContent(), /已收到|缓冲/);
    await page.locator('details').evaluate(el => { el.open = true; });
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    // Exercise real browser getUserMedia + AudioWorklet using synthetic devices.
    await page.waitForTimeout(1500);
    const micStartCount = await page.evaluate(() => window.__playback.starts);
    await page.locator('#start').click();
    await page.waitForFunction(() => parseInt(document.getElementById('inputCount').textContent) >= 3, {timeout: 90000});
    assert.ok(await page.locator('#camera').evaluate(video => video.readyState >= 2 && video.videoWidth > 0));
    await page.waitForFunction(count => window.__playback.starts > count, micStartCount);
    await page.waitForTimeout(250);
    await page.evaluate(() => {
      window.__recordMotion = true; window.__captures = [];
      window.__captureTimer = setInterval(() => window.__captures.push({at: Date.now(), image: document.getElementById('avatar').toDataURL('image/jpeg', .85)}), 40);
    });
    const beforeInterrupt = Date.now();
    const beforeStops = await page.evaluate(() => window.__playback.stops);
    await page.locator('#interrupt').click();
    await page.waitForFunction(count => window.__playback.stops > count, beforeStops);
    await page.waitForTimeout(1500);
    const transitionCapture = await page.evaluate(() => {
      clearInterval(window.__captureTimer); window.__recordMotion = false;
      return {frames: window.__captures, states: window.__motionEvents, draws: window.__motionDraws, stoppedAt: window.__playback.stoppedAt};
    });
    const interruptEvent = events.find(e => e.type === 'interrupted' && e.receivedAt >= beforeInterrupt);
    assert.ok(interruptEvent, 'No server interruption acknowledgement');
    const resumed = events.find(e => e.epoch === interruptEvent.epoch && e.transition && e.motion_resume);
    assert.ok(resumed?.motion_resume.anchor_matched, JSON.stringify(resumed));
    assert.deepEqual(resumed.motion_resume.anchor, sentInterrupts.at(-1).anchor);
    assert.ok(transitionCapture.states.some(e => e.state === 'transition'));
    assert.ok(transitionCapture.states.some(e => e.state === 'idle'));
    assert.ok(transitionCapture.draws.length > 12, 'Transition did not draw intermediate frames');
    assert.ok(transitionCapture.draws[0] - beforeInterrupt < 180, 'Visual bridge started late');
    assert.ok(transitionCapture.stoppedAt - beforeInterrupt < 180, 'Audio did not stop promptly');
    transitionCapture.frames.forEach((frame, index) => fs.writeFileSync(path.join(output, `transition-${String(index).padStart(3, '0')}.jpg`), Buffer.from(frame.image.split(',')[1], 'base64'), {flag: 'wx'}));
    const movie = spawnSync('ffmpeg', ['-nostdin', '-v', 'error', '-n', '-framerate', '25', '-i', path.join(output, 'transition-%03d.jpg'), '-an', '-c:v', 'libx264', '-threads', '4', '-pix_fmt', 'yuv420p', path.join(output, 'interruption.mp4')]);
    assert.equal(movie.status, 0, movie.stderr?.toString());
    const staleMedia = events.filter(e => ['media', 'idle'].includes(e.type) && e.receivedAt > interruptEvent.receivedAt && e.epoch < interruptEvent.epoch);
    assert.equal(staleMedia.length, 0, 'Stale video arrived after interruption');
    // A second interruption may arrive before the first transition has completed.
    await page.locator('#interrupt').click();
    await page.waitForTimeout(60);
    await page.locator('#interrupt').click();
    await page.waitForTimeout(1500);
    const latestEpoch = Math.max(...events.filter(e => e.type === 'interrupted').map(e => e.epoch));
    assert.ok(events.some(e => e.epoch === latestEpoch && e.transition && e.motion_resume?.anchor_matched));
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    await page.waitForTimeout(1500);
    // Audio input still supports avatar video, and must request no camera.
    await page.locator('#audioInput').check();
    const audioInputStartCount = await page.evaluate(() => window.__playback.starts);
    await page.locator('#start').click();
    await page.waitForFunction(count => window.__playback.starts > count, audioInputStartCount);
    const audioDevice = await page.evaluate(() => window.__devices.at(-1));
    assert.equal(audioDevice.constraints.video, false);
    assert.equal(audioDevice.videoTracks, 0);
    assert.equal(audioDevice.audioTracks, 1);
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    await page.waitForTimeout(1500);
    // Pure audio output bypasses A2V and its silent driving stream entirely.
    await page.locator('#useAvatar').uncheck();
    const healthBefore = await page.evaluate(async () => (await (await fetch('/health')).json()).avatar);
    assert.equal(healthBefore.stream_busy, false);
    const audioOutputStartCount = await page.evaluate(() => window.__playback.starts);
    const audioOnlySince = Date.now();
    await page.locator('#demo').click();
    await page.waitForFunction(count => window.__playback.starts > count, audioOutputStartCount);
    assert.equal(await page.locator('#voiceStage').isVisible(), true);
    assert.equal(await page.locator('#cameraPanel').isVisible(), false);
    assert.equal(await page.locator('#avatar').isVisible(), false);
    const audioOnlyEvents = events.filter(e => e.receivedAt >= audioOnlySince);
    assert.ok(audioOnlyEvents.some(e => e.type === 'media' && e.output_mode === 'audio'));
    assert.equal(audioOnlyEvents.filter(e => e.type === 'idle' || (e.type === 'media' && e.frames > 0)).length, 0);
    const healthAfter = await page.evaluate(async () => (await (await fetch('/health')).json()).avatar);
    assert.equal(healthAfter.render_count, healthBefore.render_count);
    await page.locator('.turn[data-role="user"] p').first().waitFor();
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.deepEqual(errors, []);
    const result = {passed: true, media_events: events.filter(e => e.type === 'media').length,
      playback_started: playback.starts, interruption_ack_ms: interruptEvent.receivedAt - beforeInterrupt,
      stale_media_after_interrupt: staleMedia.length, camera_and_microphone_worklet: true,
      decision_ms: 960, input_samples: 15360, frames_per_block: 24,
      interruption_audio_stop_ms: transitionCapture.stoppedAt - beforeInterrupt,
      interruption_first_draw_ms: transitionCapture.draws[0] - beforeInterrupt,
      transition_resume: resumed.motion_resume, repeated_interruption: true,
      transition_ms: transitionCapture.states.find(e => e.state === 'idle').at - beforeInterrupt,
      idle_animation: events.some(e => e.type === 'idle'), audio_input_without_camera: true,
      audio_output_without_avatar: true, audio_only_avatar_renders: healthAfter.render_count - healthBefore.render_count,
      user_asr_and_capture_timestamps: true, reply_generation_and_playback_timestamps: true,
      live_loudness_meter: true, postroll_drawn: await page.evaluate(() => window.__postrollDrawn),
      reconnect: true, page_errors: errors, output};
    result.separate_media_and_control = true;
    result.h264_video = true;
    result.device_labels = await page.locator('#modelDevices').textContent();
    assert.match(result.device_labels, /理解 GPU \d+.*原声 GPU \d+.*视频 GPU \d+.*字幕 GPU \d+/);
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(result, null, 2), {flag: 'wx'});
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    await page.screenshot({path: path.join(output, 'failure.png'), fullPage: true});
    console.error('Browser state:', await page.locator('body').innerText());
    console.error('Page errors:', errors); console.error(error);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();

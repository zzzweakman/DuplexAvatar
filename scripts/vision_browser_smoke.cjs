// Exercise the shipped camera capture, upload, MiniCPM vision, TTS and avatar.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const root = path.resolve(__dirname, '..');
const audioOutput = process.argv.includes('--audio-output');
const minimumDurationMs = audioOutput ? 20000 : 90000;
const questionWav = process.env.VISION_QUESTION_WAV || path.join(root, 'runtime/fixtures/vision-describe-single-zh-20261015.wav');
assert.ok(fs.existsSync(questionWav), 'Set VISION_QUESTION_WAV to a WAV asking 请描述一下你现在看到的画面。 See docs/testing.md.');
const output = path.join(root, 'runtime', `vision-browser-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(output);

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required'],
  });
  const context = await browser.newContext({ignoreHTTPSErrors: true, viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  page.setDefaultTimeout(60000);
  await page.route('**/vision-question.wav', route => route.fulfill({
    contentType: 'audio/wav', body: fs.readFileSync(questionWav),
  }));
  await context.addInitScript(() => {
    window.__visionDevices = [];
    navigator.mediaDevices.getUserMedia = async constraints => {
      const sound = new AudioContext({sampleRate: 16000});
      await sound.resume();
      const destination = sound.createMediaStreamDestination();
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
      const drawing = canvas.getContext('2d');
      let scene = 'blue', portrait, painted = 0;
      const paint = () => {
        drawing.fillStyle = scene === 'blue' ? '#1937e6' : scene === 'green' ? '#14b937' : '#eeeeee';
        drawing.fillRect(0, 0, 640, 480);
        if (scene === 'portrait' && portrait) drawing.drawImage(portrait, 80, 0, 480, 480);
        // A moving marker proves each uploaded batch contains separate captures.
        drawing.fillStyle = '#ffffff'; drawing.fillRect((painted++ * 7) % 624, 466, 16, 8);
      };
      paint(); const timer = setInterval(paint, 66);
      const camera = canvas.captureStream(15);
      const stream = new MediaStream([...destination.stream.getAudioTracks(), ...(constraints.video ? camera.getVideoTracks() : [])]);
      window.__visionDevices.push({video: !!constraints.video, stream, sound, timer});
      window.__visionInput = {
        async scene(value) {
          scene = value;
          if (value === 'portrait' && !portrait) portrait = await createImageBitmap(await (await fetch('/avatar.jpg')).blob());
          paint();
        },
        async question(url) {
          const buffer = await sound.decodeAudioData(await (await fetch(url)).arrayBuffer());
          const source = sound.createBufferSource(); source.buffer = buffer; source.connect(destination);
          source.start(sound.currentTime + .15);
          return buffer.duration;
        },
      };
      return stream;
    };
  });
  const events = [], uploads = [], errors = [], checks = [];
  let phase = 'blue';
  page.on('pageerror', error => errors.push(String(error)));
  page.on('websocket', socket => {
    socket.on('framesent', event => {
      const message = JSON.parse(event.payload);
      if (message.type === 'input') uploads.push({phase, at: Date.now(), id: message.client_input_id,
        video_frames: message.video_frames.length, video_capture_ms: message.video_capture_ms,
        video_frame_times_ms: message.video_frame_times_ms || [],
        frame_hashes: message.video_frames.map(frame => createHash('sha256').update(frame).digest('hex'))});
    });
    socket.on('framereceived', event => {
      const message = JSON.parse(event.payload);
      events.push({phase, at: Date.now(), ...Object.fromEntries(Object.entries(message).filter(([k]) => !['audio', 'frames', 'video'].includes(k)))});
    });
  });
  const completedReply = (start, expected) => {
    const replies = new Map(), ended = new Set();
    for (const event of events.slice(start)) {
      if (event.type === 'text') replies.set(event.response_id, (replies.get(event.response_id) || '') + event.text);
      if (event.type === 'response.end') ended.add(event.response_id);
    }
    return [...replies].find(([id, text]) => ended.has(id) && expected.test(text))?.[1];
  };
  const wait = async (condition, label, timeout = 45000) => {
    const start = Date.now();
    while (!condition()) {
      assert.ok(Date.now() - start < timeout, `${label}: ${JSON.stringify(events.slice(-12))}`);
      await page.waitForTimeout(150);
    }
  };
  try {
    await page.goto('https://127.0.0.1:8443', {waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    if (audioOutput) await page.locator('#useAvatar').uncheck();
    await page.locator('#start').click();
    await page.waitForFunction(() => document.getElementById('status').textContent === '会话进行中');
    const started = Date.now(), delivery = [];
    const ready = events.find(e => e.type === 'session.ready');
    assert.equal(ready.video_input_fps, 5);
    assert.equal(ready.max_video_frames_per_input, 6);
    assert.equal(ready.timing.decision_ms, audioOutput ? 1000 : 960);
    const queryScene = async (scene, expected, question, label = scene) => {
      // Keep visual recognition separate from the interruption regression: the
      // media response-end event can arrive before its audio has finished playing.
      await page.waitForFunction(() => document.getElementById('playState').textContent === '正在倾听');
      phase = label;
      const start = events.length;
      await page.evaluate(value => window.__visionInput.scene(value), scene);
      await page.evaluate(url => window.__visionInput.question(url), question);
      await wait(() => completedReply(start, expected), `Visual answer ${scene}`);
      await page.waitForFunction(() => /模型已处理 [1-9]/.test(document.getElementById('visionState').textContent));
      const consumed = events.slice(start).filter(e => e.type === 'input.processed' && e.vision_tokens > 0);
      assert.ok(consumed.length > 0, 'Need actual model vision tokens');
      assert.ok(uploads.some(e => e.phase === label && e.video_frames >= 4));
      assert.ok(consumed.every(e => e.vision_slices === e.video_frames && e.vision_tokens === e.video_frames * 64));
      assert.ok(events.slice(start).some(e => e.type === 'media' && e.valid_samples > 0), 'Original speech reached playback');
      const snapshot = await page.locator('#modelInputImage').getAttribute('src');
      assert.ok(snapshot?.startsWith('data:image/jpeg;base64,'));
      fs.writeFileSync(path.join(output, `${label}-consumed.jpg`), Buffer.from(snapshot.split(',')[1], 'base64'), {flag: 'wx'});
      checks.push({scene: label, reply: completedReply(start, expected), consumed_frames: consumed.reduce((sum, e) => sum + e.video_frames, 0),
        vision_tokens_per_frame: [...new Set(consumed.map(e => e.vision_tokens / e.video_frames))],
        feedback: await page.locator('#visionState').textContent(), capture_time: await page.locator('#visionTime').textContent()});
      console.log(label, checks.at(-1));
      await page.waitForTimeout(1200);
    };
    const scenarios = [['blue', /蓝/, '/demo/audio.wav']];
    if (!audioOutput) scenarios.push(['green', /绿/, '/demo/audio.wav'],
      ['portrait', /女|头发|上衣|衣服|项链/, '/vision-question.wav']);
    for (const scenario of scenarios) await queryScene(...scenario);
    if (!audioOutput) {
      while (Date.now() - started < 60000) await page.waitForTimeout(1000);
      await queryScene('blue', /蓝/, '/demo/audio.wav', 'after-window');
    }
    // Run beyond the one-minute point where earlier input buffering was reported.
    phase = 'sustained';
    while (Date.now() - started < minimumDurationMs) {
      delivery.push(await page.locator('#inputDelivery').evaluate(el => ({at: Date.now(), state: el.dataset.state,
        text: el.textContent, details: el.title})));
      await page.waitForTimeout(1000);
    }
    const continuous = uploads.slice(), times = continuous.flatMap(e => e.video_frame_times_ms);
    const consumed = events.filter(e => e.type === 'input.processed' && e.vision_tokens > 0);
    const intervals = times.slice(1).map((time, i) => time - times[i]).sort((a, b) => a - b);
    const processing = consumed.map(e => e.processing_ms).sort((a, b) => a - b);
    const timing = {duration_ms: Date.now() - started,
      sampled_frames: times.length, fps: (times.length - 1) * 1000 / (times.at(-1) - times[0]),
      median_capture_interval_ms: intervals[Math.floor(intervals.length / 2)],
      frames_per_batch: [...new Set(continuous.map(e => e.video_frames))].sort(),
      processing_median_ms: processing[Math.floor(processing.length / 2)],
      processing_p95_ms: processing[Math.floor(processing.length * .95)],
      max_pending_inputs: Math.max(...consumed.map(e => e.pending_inputs)),
      max_vision_context_tokens: Math.max(...consumed.map(e => e.vision_context_tokens)),
      max_vision_context_units: Math.max(...consumed.map(e => e.vision_context_units)),
      skipped_silence_inputs: consumed.at(-1).skipped_silence_inputs,
      processed_video_frames: consumed.at(-1).processed_video_frames,
      received_video_frames: consumed.at(-1).received_video_frames};
    assert.ok(timing.fps >= 4.7 && timing.fps <= 5.2, `Camera FPS: ${JSON.stringify(timing)}`);
    assert.ok(intervals.every(interval => interval > 0), 'Capture timestamps stay ordered across batches');
    assert.ok(continuous.every(e => e.video_frames <= 6));
    const batches = continuous.filter(e => e.video_frames >= 4);
    assert.ok(batches.length > (audioOutput ? 15 : 60), 'Need sustained multi-frame input');
    assert.ok(batches.filter(e => new Set(e.frame_hashes).size === e.video_frames).length / batches.length > .95,
      'Images must be independent captures, not repeated copies');
    assert.ok(consumed.every(e => e.vision_slices === e.video_frames && e.vision_tokens === e.video_frames * 64));
    assert.ok(consumed.every(e => e.last_vision_capture_ms === e.video_frame_times_ms.at(-1)));
    assert.ok(consumed.every(e => e.vision_window_mode === 'context'));
    assert.ok(timing.max_vision_context_units <= 24);
    if (!audioOutput) assert.equal(timing.max_vision_context_units, 24, 'Need to exercise context eviction');
    assert.ok(timing.max_vision_context_tokens < 12000, 'Camera context must stay bounded');
    assert.ok(timing.max_pending_inputs <= 2, `Input queue accumulated: ${JSON.stringify(timing)}`);
    assert.equal(timing.skipped_silence_inputs, 0);
    assert.ok(delivery.every(e => e.state === 'ready'), JSON.stringify(delivery));
    console.log('5 FPS sustained input', timing);
    // A visible but paused camera must not keep uploading the same stale frame.
    phase = 'paused'; await page.locator('#camera').evaluate(video => video.pause());
    await page.waitForTimeout(3600);
    assert.match(await page.locator('#visionState').textContent(), /暂无新画面/);
    assert.ok(uploads.filter(e => e.phase === 'paused').slice(-2).every(e => e.video_frames === 0));
    phase = 'resumed'; const before = events.length;
    await page.locator('#camera').evaluate(video => video.play());
    await wait(() => events.slice(before).some(e => e.type === 'input.processed' && e.vision_tokens > 0), 'Camera resumes');
    await page.locator('details').evaluate(el => { el.open = true; });
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    await page.locator('#stop').click();
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    assert.ok(await page.evaluate(() => window.__visionDevices[0].stream.getTracks().every(t => t.readyState === 'ended')));
    assert.deepEqual(errors, []);
    const result = {passed: true, checks, errors, camera_pause_detected: true, camera_resume_confirmed: true,
      same_session_scene_changes: !audioOutput, avatar_and_original_audio: !audioOutput,
      camera_with_voice_output: audioOutput, timing, delivery, events, uploads};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(result, null, 2), {flag: 'wx'});
    console.log(output);
  } catch (error) {
    fs.writeFileSync(path.join(output, 'failure.json'), JSON.stringify({error: String(error), checks, errors, events, uploads}, null, 2), {flag: 'wx'});
    throw error;
  } finally {
    await page.locator('#stop').click({timeout: 2000}).catch(() => {});
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

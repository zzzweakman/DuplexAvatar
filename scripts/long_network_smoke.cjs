// Real microphone capture and model inference through the shared-bandwidth proxy.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const seconds = Number(process.env.DURATION_SECONDS || 240);
const url = process.env.TEST_URL || 'https://127.0.0.1:8444';
const output = path.join(root, 'runtime', `long-network-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(output, {recursive: true});
const fixture = path.join(output, 'microphone.wav');
const build = spawnSync(path.join(root, '../MiniCPM-o-Demo/.venv/base/bin/python'), ['-c', `
import sys,wave,numpy as np
with wave.open(sys.argv[1]) as f:
    rate=f.getframerate(); speech=np.frombuffer(f.readframes(f.getnframes()),'<i2')
audio=np.zeros((int(sys.argv[3])+30)*rate,dtype='<i2')
for second in range(15,int(sys.argv[3]),45): audio[second*rate:second*rate+len(speech)]=speech
with wave.open(sys.argv[2],'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(audio.tobytes())
`, path.join(root, 'runtime/fixtures/question.wav'), fixture, String(seconds)], {encoding: 'utf8'});
assert.equal(build.status, 0, build.stderr);

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, env: {...process.env, CUDA_VISIBLE_DEVICES: ''},
    args: ['--no-sandbox', '--disable-gpu', '--autoplay-policy=no-user-gesture-required',
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${fixture}%noloop`],
  });
  const events = [], samples = [], errors = [], sentAt = new Map();
  let mediaBytes = 0, mediaCount = 0, lastProcessed = 0;
  try {
    const page = await browser.newPage({ignoreHTTPSErrors: true, viewport: {width: 1440, height: 1000}, permissions: ['microphone']});
    page.setDefaultTimeout(90000);
    page.on('pageerror', error => errors.push(String(error)));
    page.on('websocket', socket => {
      socket.on('framesent', event => {
        const data = JSON.parse(event.payload);
        if (data.type === 'input') {
          sentAt.set(data.client_input_id, Date.now());
          events.push({type: 'sent', id: data.client_input_id, capture_ms: data.capture_start_ms, at: Date.now()});
        }
      });
      socket.on('framereceived', event => {
        const data = JSON.parse(event.payload);
        if (data.type === 'input.received') events.push({type: 'ack', id: data.client_input_id,
          rtt_ms: Date.now() - sentAt.get(data.client_input_id), pending: data.pending_inputs, at: Date.now()});
        if (data.type === 'input.processed') {
          lastProcessed = data.processed_inputs;
          events.push({type: 'processed', count: lastProcessed, ms: data.processing_ms, at: Date.now()});
        }
        if (data.type === 'media' || data.type === 'idle') {
          assert.equal(data.video?.frame_count, 24);
          mediaBytes += Buffer.byteLength(event.payload); mediaCount++;
        }
        if (data.type === 'transcript' || data.type === 'text') events.push({type: data.type, text: data.text, at: Date.now()});
        if (data.type === 'error') errors.push(data.message);
      });
    });
    await page.goto(url, {waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.getElementById('start').disabled);
    await page.locator('#audioInput').check();
    await page.locator('#start').click();
    await page.waitForFunction(() => document.getElementById('status').textContent === '会话进行中');
    const started = Date.now();
    for (let elapsed = 0; elapsed < seconds; elapsed += 5) {
      await page.waitForTimeout(5000);
      const sample = await page.evaluate(() => ({status: document.getElementById('status').textContent,
        delivery: document.getElementById('inputDelivery').textContent, detail: document.getElementById('inputDelivery').title,
        notice: document.getElementById('notice').textContent, frames: parseInt(document.getElementById('frameCount').textContent),
        mic: document.getElementById('micState').textContent}));
      sample.elapsed = (Date.now() - started) / 1000;
      sample.local_queue = Number(sample.detail.match(/本地待发 (\d+)/)?.[1] || 0);
      sample.window = Number(sample.detail.match(/上传窗口 (\d+)/)?.[1] || 0);
      samples.push(sample);
      assert.equal(sample.status, '会话进行中', JSON.stringify(sample));
      assert.ok(!sample.mic.includes('暂缓'), JSON.stringify(sample));
      if (elapsed % 30 === 0) console.log(JSON.stringify({...sample, processed: lastProcessed}));
    }
    await page.locator('#stop').click();
    await page.waitForTimeout(6000);
    const sent = events.filter(e => e.type === 'sent');
    const first = sent[0];
    const drift = sent.map(e => ({capture_ms: e.capture_ms, lag_ms: e.at - e.capture_ms - (first.at - first.capture_ms)}));
    const finalMinute = samples.filter(s => s.elapsed > seconds - 60);
    const median = values => values.sort((a,b) => a-b)[Math.floor(values.length / 2)];
    const late = drift.filter(s => s.capture_ms > (seconds - 60) * 1000);
    const earlier = drift.filter(s => s.capture_ms >= 30000 && s.capture_ms <= 60000);
    const growth = median(late.map(s => s.lag_ms)) - median(earlier.map(s => s.lag_ms));
    assert.ok(growth < 960, `Upload lag continues growing: ${growth} ms`);
    assert.ok(Math.max(...finalMinute.map(s => s.local_queue)) <= 2, JSON.stringify(finalMinute));
    assert.ok(events.filter(e => e.type === 'transcript').length >= 3);
    assert.ok(events.filter(e => e.type === 'text').length >= 3);
    assert.ok(mediaCount >= seconds / 2);
    assert.deepEqual(errors, []);
    const result = {passed: true, seconds, url, shaped_network: {rtt_ms: 2400, down_mbps: 4, up_mbps: 1},
      inputs_sent: sent.length, inputs_processed_at_stop: lastProcessed,
      max_local_queue: Math.max(...samples.map(s => s.local_queue)),
      final_minute_max_local_queue: Math.max(...finalMinute.map(s => s.local_queue)),
      upload_lag_growth_ms_after_first_minute: growth,
      max_upload_window: Math.max(...samples.map(s => s.window)),
      input_processing_median_ms: median(events.filter(e => e.type === 'processed').map(e => e.ms)),
      media_count: mediaCount, media_mbps: mediaBytes * 8 / seconds / 1e6,
      user_transcripts: events.filter(e => e.type === 'transcript').length,
      reply_text_chunks: events.filter(e => e.type === 'text').length, page_errors: errors, output};
    for (const [name, data] of Object.entries({'results.json': result, 'samples.json': samples, 'events.json': events})) {
      fs.writeFileSync(path.join(output, name), JSON.stringify(data, null, 2), {flag: 'wx'});
    }
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    fs.writeFileSync(path.join(output, 'failure.json'), JSON.stringify({error: String(error), samples, events, errors}, null, 2), {flag: 'wx'});
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

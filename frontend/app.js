import {InputFlow} from './input-flow.mjs';
import {bridgePlan, blendWeight} from './motion-transition.mjs';
import {preferredVideoCodec, decodeVideoBlock} from './video-transport.mjs';
import {ListeningMotion} from './listening-motion.mjs';
import {CameraInput} from './camera-input.mjs';
const $ = id => document.getElementById(id);
let socket, mediaSocket, outputContext, captureContext, stream, recorder;
let epoch = 0, sessionActive = false, sessionEnding = false, serviceBusy = false, serviceReady = false, nextAudioTime = 0;
let captureParts = [], captured = 0, sentInputs = 0, displayedFrames = 0, pendingInputs = 0;
let sources = new Set(), videoFrames = [], transcriptNodes = new Map();
let mediaChain = Promise.resolve(), interruptAt = 0, voiceTicks = 0;
let idleFrames = [], idleChain = Promise.resolve(), playbackRevision = 0;
let brainReady = false, avatarReady = false, sessionSerial = 0;
let playbackSequence = 0;
let idleSequence = 0, playbackStart = 0;
const listeningMotion = new ListeningMotion();
let inputFlow, sessionStartedMs = 0, captureOrigin = 0, blockStartMs = 0;
let capturePaused = false, sourceIsMicrophone = false, inputState = {}, lastInputAt = 0;
let nextSpeechEnd = 0, responsePlayback = new Map();
let inputSamples = 16000, inputUnitMs = 1000;
let bridgeFrames = [], motionTransition = null, lastDisplayed = null;
let lastWasIdle = null;
let visionFrames = new Map(), lastCameraCaptureAt = 0, lastCameraTime = -1;
let lastVisionConfirmationAt = 0;
let cameraInput, cameraTimer, visionInputFps = 5, maxVideoFrames = 6;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const avatar = $('avatar'), drawing = avatar.getContext('2d');
const portrait = new Image(); portrait.src = '/avatar.jpg';
portrait.onload = () => drawing.drawImage(portrait, 0, 0, avatar.width, avatar.height);

function notice(message, error = false) { $('notice').textContent = message; $('notice').classList.toggle('error', error); }
function controls() {
  serviceReady = brainReady && (!$('useAvatar').checked || avatarReady);
  $('start').disabled = !serviceReady || sessionActive || sessionEnding || serviceBusy;
  $('demo').disabled = !serviceReady || sessionActive || sessionEnding || serviceBusy;
  $('stop').disabled = !sessionActive; $('interrupt').disabled = !sessionActive;
  $('useCamera').disabled = sessionActive;
  $('audioInput').disabled = sessionActive; $('useAvatar').disabled = sessionActive;
}
function modes() {
  $('cameraPanel').hidden = !$('useCamera').checked;
  $('cameraFeedback').hidden = !$('useCamera').checked;
  avatar.hidden = !$('useAvatar').checked; $('voiceStage').hidden = $('useAvatar').checked;
  $('outputLabel').textContent = $('useAvatar').checked ? 'MiniCPM 原声 · FlashHead Lite' : 'MiniCPM 原声';
  $('intro').textContent = $('useAvatar').checked
    ? ($('useCamera').checked ? '用声音交谈，也可以让她看见你的画面。倾听时，人物会保持自然的小幅动作。' : '只使用麦克风对话，数字人会用原声回复。倾听时保持自然的小幅动作。')
    : ($('useCamera').checked ? '麦克风和摄像头会参与对话，回复只通过声音播放。' : '只使用麦克风和扬声器对话，保留回复文字。');
  $('demo').textContent = $('useCamera').checked ? '运行音视频样例' : '运行音频样例';
  if (!sessionActive) notice($('useCamera').checked ? '样例使用预录音频和红色图片，无需麦克风权限。' : '仅使用音频，不申请摄像头权限，也不发送图片。');
  controls();
}
function encodePCM(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let result = ''; for (let i = 0; i < bytes.length; i += 8192) result += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(result);
}
function decodePCM(value) {
  const binary = atob(value), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
function clearPlayback() {
  playbackRevision++; clearIdle();
  listeningMotion.clear(); idleSequence = 0; playbackStart = 0;
  for (const source of sources) { try { source.stop(); } catch {} }
  sources.clear(); for (const frame of videoFrames) frame.image.close();
  videoFrames = []; nextAudioTime = 0; playbackSequence = 0;
  for (const frame of bridgeFrames) frame.image.close();
  bridgeFrames = []; motionTransition = null;
  lastWasIdle = null;
}
function clearIdle() { for (const frame of idleFrames) frame.image.close(); idleFrames = []; }
function copySilentFrame(image) {
  if (typeof image.clone === 'function') return image.clone();
  const canvas = new OffscreenCanvas(image.width, image.height);
  canvas.getContext('2d').drawImage(image, 0, 0);
  return canvas.transferToImageBitmap();
}
function startMotionTransition(nextEpoch) {
  const now = performance.now() / 1000, audioNow = outputContext?.currentTime || 0;
  const queue = bridgeFrames.length ? bridgeFrames : (videoFrames.length ? videoFrames : idleFrames);
  const fromSpeech = queue === videoFrames;
  const plan = bridgePlan(queue, fromSpeech ? audioNow : now, nextEpoch, reducedMotion.matches);
  if (fromSpeech) for (const frame of plan.frames) frame.time += now - audioNow;
  const kept = new Set(plan.original);
  bridgeFrames = bridgeFrames.filter(frame => !kept.has(frame));
  videoFrames = videoFrames.filter(frame => !kept.has(frame));
  idleFrames = idleFrames.filter(frame => !kept.has(frame));
  const anchor = plan.anchor || lastDisplayed;
  clearPlayback(); // Selected bridge frames were detached; all audio stops here.
  bridgeFrames = plan.frames;
  if ($('useAvatar').checked) {
    motionTransition = {epoch: nextEpoch, until: bridgeFrames.at(-1)?.time || now, blendStart: null};
    avatar.dataset.motion = 'transition';
  }
  return anchor;
}
function interrupt() {
  if (!sessionActive || socket?.readyState !== WebSocket.OPEN) return;
  const anchor = startMotionTransition(epoch + 1);
  epoch += 1; interruptAt = performance.now(); voiceTicks = 0; nextSpeechEnd = 0;
  $('playState').textContent = '已打断，正在倾听';
  socket.send(JSON.stringify({type: 'interrupt', anchor}));
}
function drawFrames() {
  const now = performance.now() / 1000, audioNow = outputContext?.currentTime || 0;
  const speaking = outputContext && audioNow >= playbackStart && nextAudioTime > audioNow;
  const queue = bridgeFrames.length ? bridgeFrames : (videoFrames[0]?.time <= audioNow ? videoFrames : (!speaking ? idleFrames : []));
  const queueNow = queue === videoFrames ? audioNow : now;
  if (outputContext && sessionActive) {
    let latest;
    while (queue.length && queue[0].time <= queueNow) {
      if (latest) latest.image.close(); latest = queue.shift();
    }
    // A queued frame in the future is normal 25 FPS pacing, not starvation.
    // Do not reverse cached motion in the short gap before that frame is due.
    if (!latest && !speaking && !bridgeFrames.length && !idleFrames.length && !videoFrames.length && !reducedMotion.matches) {
      latest = listeningMotion.next(now, epoch);
    }
    if (latest) {
      if (latest.epoch === epoch) {
        const switchingMode = lastWasIdle !== null && lastWasIdle !== !!latest.idle;
        const blockBoundary = lastDisplayed && lastDisplayed.sequence !== latest.sequence;
        if (!latest.bridge && !latest.fallback && !motionTransition &&
            (switchingMode || blockBoundary || latest.idle && listeningMotion.active)) {
          // Re-encoding motion context can leave a small block-edge pose change.
          // Smooth only this visual edge; original audio and frame timing stay put.
          motionTransition = {epoch, until: now, blendStart: null,
            duration: switchingMode || listeningMotion.active ? .12 : .08};
        }
        if (motionTransition && !latest.bridge) {
          if (motionTransition.blendStart === null) {
            motionTransition.from = document.createElement('canvas'); motionTransition.to = document.createElement('canvas');
            for (const canvas of [motionTransition.from, motionTransition.to]) { canvas.width = avatar.width; canvas.height = avatar.height; }
            motionTransition.from.getContext('2d').drawImage(avatar, 0, 0);
            motionTransition.blendStart = now;
          }
          motionTransition.to.getContext('2d').drawImage(latest.image, 0, 0, avatar.width, avatar.height);
          motionTransition.targetMotion = latest.idle ? 'idle' : 'speech';
        } else drawing.drawImage(latest.image, 0, 0, avatar.width, avatar.height);
        lastDisplayed = {sequence: latest.sequence, frame_index: latest.frame_index};
        if (!latest.bridge) lastWasIdle = !!latest.idle;
        displayedFrames++; $('frameCount').textContent = `${displayedFrames} 帧`;
        avatar.dataset.motion = motionTransition ? 'transition' : latest.idle ? 'idle' : (latest.postroll ? 'postroll' : 'speech');
        avatar.dataset.videoSource = latest.fallback ? 'buffered-idle' : 'live';
        if (latest.idle && !reducedMotion.matches) listeningMotion.remember(latest, now, copySilentFrame);
        else if (!latest.bridge) listeningMotion.clear();
      }
      if (!latest.fallback) latest.image.close();
    }
    if (speaking) {
      while (idleFrames.length && idleFrames[0].sequence < playbackSequence) idleFrames.shift().image.close();
      // Audio devices can stall their clock. Keep prefetched post-speech idle
      // after the actual audio endpoint, without freezing idle on that clock.
      if (idleFrames.length && idleFrames[0].time < now + nextAudioTime - audioNow) {
        const shift = now + nextAudioTime - audioNow - idleFrames[0].time;
        for (const frame of idleFrames) frame.time += shift;
      }
    }
  }
  if (motionTransition?.blendStart != null) {
    const alpha = blendWeight(now - motionTransition.blendStart,
      reducedMotion.matches ? .08 : motionTransition.duration || .12);
    drawing.drawImage(motionTransition.from, 0, 0);
    drawing.globalAlpha = alpha; drawing.drawImage(motionTransition.to, 0, 0); drawing.globalAlpha = 1;
    if (alpha >= 1) {
      avatar.dataset.motion = motionTransition.targetMotion;
      motionTransition = null;
      if (!speaking) $('playState').textContent = '正在倾听';
    }
  }
  if (!motionTransition && sessionActive && outputContext && nextAudioTime && outputContext.currentTime > nextAudioTime) {
    $('playState').textContent = '正在倾听';
  } else if (!motionTransition && sessionActive && outputContext && outputContext.currentTime >= nextSpeechEnd && nextAudioTime > outputContext.currentTime) {
    $('playState').textContent = '句尾动作';
  }
  requestAnimationFrame(drawFrames);
}
requestAnimationFrame(drawFrames);

async function decodeFrames(message) {
  return decodeVideoBlock(message);
}

async function playIdle(message) {
  const context = outputContext, revision = playbackRevision;
  if (!context || !sessionActive || message.epoch !== epoch || message.sequence <= playbackSequence) return;
  const frames = await decodeFrames(message);
  if (!sessionActive || context !== outputContext || message.epoch !== epoch || revision !== playbackRevision) {
    frames.forEach(frame => frame.close()); return;
  }
  const now = performance.now() / 1000;
  const begin = Math.max(now + .02, now + Math.max(0, nextAudioTime - context.currentTime), (idleFrames.at(-1)?.time ?? -1) + 1 / message.fps,
    (motionTransition?.until ?? -1) + 1 / message.fps);
  if (begin - now > 2) { frames.forEach(frame => frame.close()); return; }
  idleFrames.push(...frames.map((image, index) => ({image, epoch, idle: true, time: begin + index / message.fps,
    sequence: message.sequence, frame_index: index, fps: message.fps})));
  idleSequence = message.sequence;
  if (nextAudioTime <= context.currentTime && !motionTransition) $('playState').textContent = '正在倾听';
}

async function playMedia(message) {
  const context = outputContext, expectedEpoch = message.epoch;
  if (!context || !sessionActive || expectedEpoch !== epoch) return;
  const frames = await decodeFrames(message);
  if (!sessionActive || context !== outputContext || epoch !== expectedEpoch) { frames.forEach(f => f.close()); return; }
  const pcm = decodePCM(message.audio);
  if (pcm.length && motionTransition) {
    for (const frame of bridgeFrames) frame.image.close();
    bridgeFrames = []; motionTransition.until = performance.now() / 1000;
  }
  const start = Math.max(context.currentTime + 0.08, nextAudioTime);
  if (start - context.currentTime > 5) { frames.forEach(f => f.close()); throw new Error('播放积压超过 5 秒，请结束后重试。'); }
  if (pcm.length) {
    const buffer = context.createBuffer(1, pcm.length, message.sample_rate); buffer.copyToChannel(pcm, 0);
    const source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination);
    sources.add(source); source.onended = () => sources.delete(source); source.start(start);
    nextSpeechEnd = start + pcm.length / message.sample_rate;
    if (!responsePlayback.has(message.response_id)) {
      responsePlayback.set(message.response_id, Date.now() + (start - context.currentTime) * 1000);
      updateReplyPlayback(message.response_id);
    }
  }
  if (nextAudioTime <= context.currentTime) playbackStart = start;
  nextAudioTime = start + (message.duration_samples ?? pcm.length) / message.sample_rate;
  playbackSequence = message.sequence;
  // Keep silent frames until the reply actually starts. Discard only overlapping
  // future idle frames; reception and decoding must not freeze the current pose.
  const visualStart = performance.now() / 1000 + start - context.currentTime;
  idleFrames = idleFrames.filter(frame => { if (frame.time < visualStart) return true; frame.image.close(); return false; });
  if (frames.length && avatar.width !== message.width) avatar.width = message.width;
  if (frames.length && avatar.height !== message.height) avatar.height = message.height;
  frames.forEach((image, index) => videoFrames.push({image, time: start + index / message.fps, epoch,
    sequence: message.sequence, frame_index: index, postroll: index / message.fps >= pcm.length / message.sample_rate}));
  $('playState').textContent = pcm.length ? '正在回复' : '句尾动作';
  $('rtf').textContent = message.output_mode === 'audio' ? '原声直出' : `${message.rtf.toFixed(2)} × 实时耗时`;
}

function clockTime(ms) { return new Date(ms).toLocaleTimeString('zh-CN', {hour12: false}); }
function relativeTime(ms) {
  const seconds = Math.max(0, ms) / 1000;
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
}
function turnNode(key, role, timestamp, label) {
  if (!transcriptNodes.has(key)) {
    $('transcript').querySelector('.empty')?.remove();
    const node = document.createElement('article'); node.className = 'turn'; node.dataset.role = role;
    node.dataset.order = timestamp;
    const meta = document.createElement('div'); meta.className = 'turn-meta';
    const speaker = document.createElement('span'); speaker.className = 'turn-role'; speaker.textContent = role === 'user' ? '你 · 语音识别' : 'MiniCPM';
    const time = document.createElement('time'); time.dateTime = new Date(timestamp).toISOString(); time.textContent = label;
    meta.append(speaker, time);
    const p = document.createElement('p'); node.append(meta, p);
    const after = [...$('transcript').children].find(item => Number(item.dataset.order) > timestamp);
    $('transcript').insertBefore(node, after || null); transcriptNodes.set(key, node);
  }
  return transcriptNodes.get(key);
}
function updateReplyPlayback(key) {
  const node = transcriptNodes.get(key), timestamp = responsePlayback.get(key);
  if (!node || !timestamp) return;
  let played = node.querySelector('.turn-played');
  if (!played) { played = document.createElement('time'); played.className = 'turn-played'; node.append(played); }
  played.dateTime = new Date(timestamp).toISOString(); played.textContent = `播放 ${clockTime(timestamp)}`;
}
function textDelta(message) {
  if (message.epoch !== epoch) return;
  const key = message.response_id || 'reply', timestamp = message.generated_at_ms || Date.now();
  const node = turnNode(key, 'assistant', timestamp, `生成 ${clockTime(timestamp)}`);
  node.querySelector('p').textContent += message.text; updateReplyPlayback(key);
  $('transcript').scrollTop = $('transcript').scrollHeight;
}
function userTranscript(message) {
  const timestamp = message.captured_at_ms;
  const node = turnNode(message.utterance_id, 'user', timestamp,
    `${clockTime(timestamp)} · ${relativeTime(message.start_ms)}–${relativeTime(message.end_ms)}`);
  node.querySelector('p').textContent = message.text;
  $('transcript').scrollTop = $('transcript').scrollHeight;
}

function inputLevel(pcm) {
  const energy = Math.sqrt(pcm.reduce((sum, value) => sum + value * value, 0) / pcm.length);
  const db = Math.max(-60, 20 * Math.log10(Math.max(energy, 1e-8)));
  $('inputDb').textContent = `${db <= -60 ? '≤ −60' : db.toFixed(1)} dBFS`;
  $('inputMeter').setAttribute('aria-valuenow', db.toFixed(1));
  $('inputMeter').dataset.level = db > -6 ? 'loud' : 'normal';
  $('meterFill').style.width = `${(db + 60) / 60 * 100}%`;
  $('micState').textContent = !sourceIsMicrophone ? '样例音频' : db > -6 ? '声音较大' : db > -44 ? '正在收到声音' : '麦克风已开启 · 音量较低';
  lastInputAt = performance.now();
  return energy;
}

function inputProgress(message = inputState) {
  inputState = message;
  const unseen = inputFlow ? [...inputFlow.inflight.keys()].filter(id => id > (message.last_received_client_input_id || 0)).length : 0;
  const uploadPending = (inputFlow?.queue.length || 0) + unseen;
  pendingInputs = (message.pending_inputs || 0) + uploadPending;
  const pendingSeconds = pendingInputs * inputUnitMs / 1000;
  const queuedSeconds = ((message.pending_inputs || 0) + (inputFlow?.queue.length || 0)) * inputUnitMs / 1000;
  const receivedSeconds = Number(((message.received_audio_ms ?? (message.received_inputs || 0) * inputUnitMs) / 1000).toFixed(2));
  $('inputDelivery').dataset.state = queuedSeconds > 2 ? 'buffering' : 'ready';
  $('inputDelivery').textContent = message.received_inputs
    ? (queuedSeconds > 2 ? `输入缓冲 ${Number(queuedSeconds.toFixed(2))} 秒 · ${(inputFlow?.queue.length || 0) > (message.pending_inputs || 0) ? '等待上传' : '模型正在处理'}` : `服务已收到 ${receivedSeconds} 秒音频`)
    : (inputFlow?.inflight.size ? '音频发送中' : '等待音频输入');
  $('inputDelivery').title = `上传往返约 ${Math.round(inputFlow?.rttMs || 0)} ms；上传窗口 ${inputFlow?.window || 2} 块；本地待发 ${inputFlow?.queue.length || 0} 块。`;
  $('silenceSkipped').textContent = `${Number(((message.skipped_silence_ms ?? (message.skipped_silence_inputs || 0) * inputUnitMs) / 1000).toFixed(2))} 秒`;
  if (message.processing_ms != null) $('inputLatency').textContent = `${message.processing_ms.toFixed(0)} ms / ${inputUnitMs} ms 音频`;
  if (sourceIsMicrophone && captureContext && !capturePaused && pendingSeconds >= 24) {
    capturePaused = true; captureContext.suspend().catch(() => {});
    $('micState').textContent = '麦克风暂缓 · 等待缓冲消化';
    notice('正在处理已录音频，麦克风暂时暂停。请稍候，恢复后再继续说话。');
  } else if (capturePaused && pendingSeconds <= 6 && captureContext) {
    capturePaused = false; captureContext.resume().catch(() => {});
    notice('输入已恢复，可以继续说话。');
  }
}

function visionProgress(message = inputState) {
  if (!$('useCamera').checked) return;
  const id = message.last_vision_client_input_id;
  const frame = visionFrames.get(id);
  if (frame) {
    $('modelInputImage').src = `data:image/jpeg;base64,${frame}`;
    $('modelInputImage').hidden = false;
    lastVisionConfirmationAt = performance.now();
    for (const key of visionFrames.keys()) { if (key <= id) visionFrames.delete(key); }
  }
  const processed = message.processed_video_frames || 0;
  $('visionDetails').textContent = `${visionInputFps} 帧/秒 · 收到 ${message.received_video_frames || 0} 帧 · 模型已处理 ${processed} 帧`;
  if (message.vision_tokens > 0) $('visionDetails').title = `本批 ${message.vision_tokens} 个视觉 token，${message.vision_slices} 个视觉切片。`;
  const track = stream?.getVideoTracks()[0];
  let state = 'ready', label;
  if (sourceIsMicrophone && (!track || track.readyState === 'ended' || !track.enabled)) {
    state = 'error'; label = '摄像头已断开 · 请结束后重新开始';
  } else if (sourceIsMicrophone && (track.muted || $('camera').readyState < 2 ||
             lastCameraCaptureAt && performance.now() - lastCameraCaptureAt > 2500)) {
    state = 'waiting'; label = '摄像头暂无新画面 · 请检查设备';
  } else if (processed && performance.now() - lastVisionConfirmationAt < 3500) {
    label = `模型已处理 ${processed} 帧画面`;
  } else if (message.received_video_frames) {
    state = 'waiting'; label = '画面已上传 · 等待模型处理';
  } else {
    state = 'waiting'; label = '摄像头已开启 · 等待画面上传';
  }
  $('visionState').textContent = label; $('cameraFeedback').dataset.state = state;
  if (message.last_vision_capture_ms != null) {
    const timestamp = sessionStartedMs + message.last_vision_capture_ms;
    $('visionTime').dateTime = new Date(timestamp).toISOString();
    $('visionTime').textContent = `采集 ${clockTime(timestamp)}`;
  }
}

function cameraFrame() {
  const track = stream?.getVideoTracks()[0];
  if (!track || track.readyState !== 'live' || !track.enabled || track.muted || $('camera').readyState < 2) return null;
  const canvas = $('capture'), video = $('camera');
  if (video.paused || !video.videoWidth || !video.videoHeight || video.currentTime === lastCameraTime) return null;
  lastCameraTime = video.currentTime; lastCameraCaptureAt = performance.now();
  canvas.width = 448; canvas.height = Math.round(video.videoHeight / video.videoWidth * 448);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  $('cameraHint').hidden = true;
  return {image: canvas.toDataURL('image/jpeg', 0.75).split(',')[1],
    capturedAt: Math.max(0, lastCameraCaptureAt - captureOrigin)};
}
function startCameraCapture(isCurrent) {
  cameraInput = new CameraInput(visionInputFps, maxVideoFrames);
  cameraInput.reset(performance.now());
  const collect = () => {
    if (!isCurrent() || !cameraInput) return;
    const now = performance.now();
    if (capturePaused || captureContext?.state !== 'running') cameraInput.reset(now);
    else cameraInput.sample(now, cameraFrame);
    cameraTimer = setTimeout(collect, Math.max(4, cameraInput.nextAt - performance.now()));
  };
  collect();
}
function sendInput(pcm, frames = [], captureStartMs = Math.max(0, performance.now() - captureOrigin - inputUnitMs), frameTimes) {
  if (!sessionActive || socket?.readyState !== WebSocket.OPEN) return;
  try {
    const item = {type: 'input', audio: encodePCM(pcm), video_frames: frames, capture_start_ms: captureStartMs};
    if (frames.length) {
      item.video_frame_times_ms = frameTimes || frames.map(() => captureStartMs);
      item.video_capture_ms = item.video_frame_times_ms.at(-1);
    }
    inputFlow.enqueue(item);
    if (frames.length) visionFrames.set(item.client_input_id, frames.at(-1));
    while (visionFrames.size > 64) visionFrames.delete(visionFrames.keys().next().value);
    sentInputs++; $('inputCount').textContent = `${sentInputs} 段`; inputProgress();
  } catch (error) { notice(error.message, true); stop(); }
}

async function start(useMicrophone) {
  if (sessionActive || sessionEnding || serviceBusy) return;
  const serial = ++sessionSerial, isCurrent = () => sessionActive && serial === sessionSerial;
  let useVideo = $('useCamera').checked;
  const useAvatar = $('useAvatar').checked;
  sessionActive = true; epoch = 0; sentInputs = 0; pendingInputs = 0; displayedFrames = 0;
  inputFlow = null; inputState = {}; capturePaused = false; sourceIsMicrophone = useMicrophone;
  responsePlayback = new Map(); nextSpeechEnd = 0; lastInputAt = 0;
  lastDisplayed = null; inputSamples = 16000; inputUnitMs = 1000;
  visionFrames.clear(); lastCameraCaptureAt = 0; lastCameraTime = -1; lastVisionConfirmationAt = 0;
  $('visionState').textContent = useVideo ? '正在准备画面输入' : '仅音频输入';
  $('cameraFeedback').dataset.state = 'waiting'; $('visionTime').textContent = '';
  $('visionDetails').textContent = '尚未处理画面'; $('visionDetails').title = '';
  $('modelInputImage').hidden = true; $('modelInputImage').removeAttribute('src');
  $('inputCount').textContent = '0 段'; $('frameCount').textContent = '0 帧'; $('rtf').textContent = '—';
  captureParts = []; captured = 0; mediaChain = Promise.resolve(); idleChain = Promise.resolve(); transcriptNodes = new Map();
  $('transcript').replaceChildren(); controls(); $('status').textContent = '正在建立会话';
  notice(useMicrophone ? '正在请求设备权限…' : '正在准备音视频样例…');
  try {
    outputContext = new AudioContext({sampleRate: 24000, latencyHint: 'interactive'});
    await outputContext.resume();
    if (!isCurrent()) return;
    if (useMicrophone) {
      try {
        const requested = await navigator.mediaDevices.getUserMedia({
          audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true},
          video: useVideo ? {width: 640, height: 480, frameRate: 15} : false,
        });
        if (!isCurrent()) { requested.getTracks().forEach(track => track.stop()); return; }
        stream = requested;
      } catch (error) {
        if (!isCurrent()) return;
        if (!useVideo) throw error;
        const requested = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}, video: false});
        if (!isCurrent()) { requested.getTracks().forEach(track => track.stop()); return; }
        stream = requested;
        useVideo = false; $('audioInput').checked = true; modes();
        notice('摄像头不可用，已使用语音模式。');
      }
      $('camera').srcObject = stream; $('cameraHint').textContent = '正在等待摄像头画面';
      $('cameraHint').hidden = !useVideo;
      if (useVideo) $('camera').play().catch(() => {
        if (isCurrent()) { $('cameraHint').hidden = false; $('cameraHint').textContent = '预览未播放，请结束后重新开始'; }
      });
    }
    const videoCodec = useAvatar ? await preferredVideoCodec() : 'jpeg';
    if (!isCurrent()) return;
    const connection = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    socket = connection;
    await new Promise((resolve, reject) => {
      let ready = false;
      let mediaAck = false;
      const acknowledgeMedia = message => {
        if (mediaAck && isCurrent() && connection.readyState === WebSocket.OPEN) {
          connection.send(JSON.stringify({type: 'media.ack', sequence: message.sequence}));
        }
      };
      connection.onopen = () => connection.send(JSON.stringify({type: 'session.start',
        input_mode: useVideo ? 'audiovisual' : 'audio', output_mode: useAvatar ? 'avatar' : 'audio',
        media_channel: 'separate', media_ack: true, video_codec: videoCodec}));
      connection.onerror = () => { if (!ready) reject(new Error('无法连接服务。')); };
      connection.onmessage = event => {
        if (!isCurrent() || socket !== connection) return;
        const message = JSON.parse(event.data);
        if (message.type === 'session.ready') {
          ready = true; epoch = message.epoch; sessionStartedMs = message.session_started_ms || Date.now(); captureOrigin = performance.now();
          mediaAck = message.media_ack === true;
          inputSamples = message.timing?.input_samples || 16000; inputUnitMs = message.timing?.decision_ms || 1000;
          visionInputFps = message.video_input_fps || 5; maxVideoFrames = message.max_video_frames_per_input || 6;
          $('blockClock').textContent = message.timing?.frames_per_block
            ? `${inputUnitMs} ms · ${message.timing.frames_per_block} 帧 / 块` : `${inputUnitMs} ms · 纯音频`;
          inputFlow = new InputFlow(item => connection.send(JSON.stringify(item)),
            () => connection.readyState === WebSocket.OPEN && connection.bufferedAmount < 1_000_000, message.input_window || 2,
            Math.ceil((message.input_buffer_seconds || 30) * 1000 / inputUnitMs),
            {unitMs: inputUnitMs, maxWindow: message.input_window_max || 8});
          if (message.media_channel) {
            const media = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${message.media_channel.path}`);
            mediaSocket = media;
            const deadline = setTimeout(() => reject(new Error('视频连接建立超时，请重新开始。')), 12000);
            media.onopen = () => media.send(JSON.stringify({session_id: message.session_id, token: message.media_channel.token}));
            media.onmessage = event => {
              if (!isCurrent() || mediaSocket !== media) return;
              if (JSON.parse(event.data).type === 'media.ready') { clearTimeout(deadline); resolve(); }
              else connection.onmessage(event);
            };
            media.onerror = () => { clearTimeout(deadline); reject(new Error('无法建立视频连接。')); };
            media.onclose = () => {
              clearTimeout(deadline);
              if (isCurrent() && mediaSocket === media) { reject(new Error('视频连接已断开。')); notice('视频连接已断开，请重新开始。', true); stop(); }
            };
          } else resolve();
        }
        else if (message.type === 'text') textDelta(message);
        else if (message.type === 'transcript') userTranscript(message);
        else if (message.type === 'asr.status') {
          $('asrState').dataset.state = message.state;
          $('asrState').textContent = message.state === 'processing' ? '正在识别语音' : message.state === 'ready' ? '字幕就绪' : '字幕暂不可用';
          $('asrState').title = message.message || '';
        }
        else if (message.type === 'media') {
          if (message.epoch !== epoch) { acknowledgeMedia(message); return; }
          playbackRevision++;
          mediaChain = mediaChain.then(() => { if (socket === connection) return playMedia(message); })
            .then(() => acknowledgeMedia(message)).catch(error => { if (isCurrent()) { notice(error.message, true); stop(); } });
        } else if (message.type === 'idle') {
          // A small idle block may decode faster than the speech/postroll block
          // before it. Wait until that earlier block has reserved its timeline.
          idleChain = Promise.all([idleChain, mediaChain]).then(() => { if (socket === connection) return playIdle(message); })
            .then(() => acknowledgeMedia(message)).catch(error => { if (isCurrent()) { notice(error.message, true); stop(); } });
        } else if (message.type === 'interrupted' && message.epoch > epoch) {
          startMotionTransition(message.epoch); epoch = message.epoch; nextSpeechEnd = 0;
        }
        else if (message.type === 'input.received') { inputFlow?.received(message.client_input_id); inputProgress(message); visionProgress(message); }
        else if (message.type === 'input.processed') { inputProgress(message); visionProgress(message); }
        else if (message.type === 'error') { notice(message.message, true); if (!ready) reject(new Error(message.message)); else stop(); }
      };
      connection.onclose = () => {
        if (!ready) reject(new Error('会话建立失败，请检查服务日志。'));
        if (socket === connection && sessionActive) { notice('连接已结束。'); stop(false); }
      };
    });
    if (!isCurrent()) return;
    $('status').textContent = '会话进行中'; $('playState').textContent = '正在倾听';
    if (useMicrophone) {
      captureContext = new AudioContext({sampleRate: 16000});
      await captureContext.audioWorklet.addModule('/static/capture-worklet.js');
      if (!isCurrent()) return;
      await captureContext.resume();
      lastInputAt = performance.now(); $('micState').textContent = '麦克风已开启 · 等待声音';
      const source = captureContext.createMediaStreamSource(stream);
      recorder = new AudioWorkletNode(captureContext, 'pcm-collector');
      const muted = captureContext.createGain(); muted.gain.value = 0;
      source.connect(recorder); recorder.connect(muted); muted.connect(captureContext.destination);
      if (useVideo) startCameraCapture(isCurrent);
      recorder.port.onmessage = event => {
        if (!isCurrent()) return;
        const pcm = event.data;
        const energy = inputLevel(pcm);
        const speaking = outputContext && nextSpeechEnd > outputContext.currentTime;
        voiceTicks = speaking && energy > 0.04 ? voiceTicks + pcm.length / 16 : 0;
        if (voiceTicks >= 200 && performance.now() - interruptAt > 1200) interrupt();
        if (!captured) blockStartMs = Math.max(0, performance.now() - captureOrigin - pcm.length / 16);
        captureParts.push(pcm); captured += pcm.length;
        if (captured >= inputSamples) {
          const block = new Float32Array(captured); let offset = 0;
          for (const part of captureParts) { block.set(part, offset); offset += part.length; }
          const frames = cameraInput?.drain() || [];
          sendInput(block.slice(0, inputSamples), frames.map(frame => frame.image), blockStartMs, frames.map(frame => frame.capturedAt));
          blockStartMs += inputUnitMs;
          const tail = block.slice(inputSamples); captureParts = tail.length ? [tail] : []; captured = tail.length;
        }
      };
      notice('可以开始说话。插话会暂停当前回复，也可以点击“打断当前回复”。');
    }
  } catch (error) { if (isCurrent()) { notice(error.message, true); await stop(); } throw error; }
}

async function stop(send = true) {
  if (sessionEnding) return;
  sessionEnding = true; serviceBusy = true;
  sessionSerial++;
  sessionActive = false; clearPlayback();
  clearTimeout(cameraTimer); cameraTimer = null; cameraInput = null;
  $('status').textContent = '正在结束会话'; controls();
  inputFlow = null; capturePaused = false; nextSpeechEnd = 0;
  $('micState').textContent = '麦克风未开启'; $('inputDb').textContent = '— dBFS'; $('meterFill').style.width = '0%';
  $('inputMeter').setAttribute('aria-valuenow', '-60'); $('inputDelivery').textContent = '录音已结束';
  $('inputDelivery').dataset.state = 'ready';
  const retiringControl = socket;
  const closed = new Promise(resolve => {
    if (!retiringControl || retiringControl.readyState === WebSocket.CLOSED) { resolve(); return; }
    const timer = setTimeout(() => { retiringControl.close(); resolve(); }, 10000);
    retiringControl.addEventListener('close', () => { clearTimeout(timer); resolve(); }, {once: true});
    if (retiringControl.readyState === WebSocket.OPEN && send) {
      retiringControl.send(JSON.stringify({type: 'session.stop'}));
    } else retiringControl.close();
  });
  socket = null; stream?.getTracks().forEach(track => track.stop()); stream = null;
  const retiringMedia = mediaSocket; mediaSocket = null;
  // Let the server observe session.stop before this independent socket closes.
  // The server normally closes it immediately; the timer covers a lost control link.
  if (retiringMedia) setTimeout(() => retiringMedia.close(), 5000);
  $('camera').srcObject = null; $('cameraHint').hidden = false; $('cameraHint').textContent = '摄像头未开启';
  visionFrames.clear(); $('visionState').textContent = '画面输入已结束';
  $('cameraFeedback').dataset.state = 'ready';
  recorder?.disconnect(); recorder = null;
  const oldCapture = captureContext, oldOutput = outputContext; captureContext = null; outputContext = null;
  await Promise.allSettled([oldCapture?.close(), oldOutput?.close(), closed]);
  sessionEnding = false;
  await pollHealth();
  $('playState').textContent = '等待开始';
  if (send && !$('notice').classList.contains('error')) notice('会话已结束。可切换输入和输出方式，再开始对话。');
  controls();
}

async function demo() {
  let demoConnection;
  try {
    await start(false);
    demoConnection = socket;
    const demoContext = outputContext;
    const isCurrent = () => sessionActive && socket === demoConnection && outputContext === demoContext;
    const data = await (await fetch('/demo/audio.wav')).arrayBuffer();
    if (!isCurrent()) return;
    const decoded = await demoContext.decodeAudioData(data);
    if (!isCurrent()) return;
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start();
    const pcm = (await offline.startRendering()).getChannelData(0);
    if (!isCurrent()) return;
    const image = document.createElement('canvas'); image.width = 224; image.height = 224;
    image.getContext('2d').fillStyle = '#ed2929'; image.getContext('2d').fillRect(0, 0, 224, 224);
    const frame = image.toDataURL('image/jpeg').split(',')[1];
    const frames = $('useCamera').checked ? [frame] : [];
    notice(frames.length ? '正在输入预录问题和红色图片，随后等待回复。' : '正在输入预录问题，随后等待语音回复。');
    const total = Math.ceil(pcm.length / inputSamples) + Math.ceil(18000 / inputUnitMs);
    const demoStarted = performance.now();
    for (let index = 0; index < total && isCurrent(); index++) {
      const block = new Float32Array(inputSamples); block.set(pcm.subarray(index * inputSamples, (index + 1) * inputSamples));
      inputLevel(block); sendInput(block, frames, index * inputUnitMs);
      await new Promise(resolve => setTimeout(resolve, Math.max(0, demoStarted + (index + 1) * inputUnitMs - performance.now())));
    }
    if (isCurrent()) notice('样例输入完成。可结束会话，或等待剩余回复播放完毕。');
  } catch (error) { if (!demoConnection || socket === demoConnection) notice(error.message, true); }
}

async function pollHealth() {
  try {
    const result = await (await fetch('/health')).json();
    serviceBusy = result.busy === true;
    brainReady = result.audio_ready ?? result.status === 'ready'; avatarReady = result.avatar?.status === 'ready';
    serviceReady = brainReady && (!$('useAvatar').checked || avatarReady);
    const acceleration = result.brain?.acceleration;
    $('modelDevices').textContent = `理解 GPU ${acceleration?.gpu ?? '—'} / 原声 GPU ${acceleration?.tts_gpu ?? acceleration?.gpu ?? '—'} / 视频 GPU ${result.avatar?.gpu ?? '—'} / 字幕 GPU ${result.asr?.gpu ?? '—'}`;
    $('inputLatencyLabel').textContent = acceleration?.parallel_voice ? '输入决策耗时' : '输入耗时';
    $('inputLatency').title = acceleration?.parallel_voice ? '输入编码与文字决策耗时；原声合成在另一张 GPU 异步进行。' : '本段输入的模型处理耗时。';
    $('compileState').textContent = `MiniCPM ${result.brain?.acceleration?.compiled ? '已编译' : '未编译'} / FlashHead ${result.avatar?.compile ? '已编译' : '未编译'}`;
    if (!sessionActive) {
      const ready = ['ok', 'ready'].includes(result.asr?.status);
      $('asrState').textContent = ready ? '字幕就绪' : '字幕准备中'; $('asrState').dataset.state = ready ? 'ready' : 'loading';
    }
    if (!sessionActive) $('status').textContent = sessionEnding ? '正在结束会话' : serviceReady ? (serviceBusy ? '已有会话运行中' : '服务就绪') : '模型加载与预热中';
  } catch { serviceReady = false; brainReady = false; avatarReady = false; if (!sessionActive) $('status').textContent = '服务暂不可用'; }
  controls();
}
$('start').onclick = () => start(true).catch(() => {});
$('stop').onclick = () => stop(); $('interrupt').onclick = interrupt; $('demo').onclick = demo;
for (const id of ['audioInput', 'useCamera', 'useAvatar']) $(id).onchange = modes;
setInterval(() => {
  if (sessionActive && outputContext && socket?.readyState === WebSocket.OPEN) {
    inputFlow?.pump();
    if (sourceIsMicrophone && $('useCamera').checked) visionProgress();
    if (sourceIsMicrophone && !capturePaused && lastInputAt && performance.now() - lastInputAt > 2000) {
      $('micState').textContent = '尚未收到音频 · 请检查麦克风'; $('meterFill').style.width = '0%';
    }
    socket.send(JSON.stringify({type: 'playback', epoch, sequence: playbackSequence,
      speech_remaining_ms: Math.max(0, (nextAudioTime - outputContext.currentTime) * 1000),
      idle_sequence: idleSequence,
      idle_remaining_ms: Math.max(0, ((idleFrames.at(-1)?.time ?? 0) + (idleFrames.length ? 1 / idleFrames.at(-1).fps : 0) - performance.now() / 1000) * 1000)}));
  }
}, 250);
window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));
modes(); pollHealth(); setInterval(pollHealth, 5000);

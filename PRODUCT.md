# DuplexAvatar

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Plain HTML/CSS/JavaScript, Python FastAPI services, existing isolated model environments. The accepted project plan establishes a local GPU application with one HTTPS browser entry.

## Users

Developers running live audiovisual conversations on a GPU server through localhost port forwarding.

## Product Purpose

MiniCPM understands microphone and camera input, generates speech, and drives an avatar model with that same speech.

## Capabilities and Constraints

The default deployment uses physical GPUs 0–3: GPU 0 runs MiniCPM understanding, GPU 2 its original speech synthesis, GPU 1 the avatar, and GPU 3 independent subtitles. Other deployments must explicitly configure physical GPU assignments and their allowed device list; a 4–7 example preserves the development server's measured configuration. One selected A2V backend at a time. The integrated backend is FlashHead Lite. Preserve MiniCPM speech. Prioritize live interaction and interruption. FlashHead Pro, FlashTalk and LiveAvatar are future possibilities; do not advertise unimplemented backends as available.

## Operating Context

Desktop browser, microphone/camera permissions, self-signed localhost HTTPS. The avatar and camera are different streams. Model readiness and errors must be visible.

Input mode (audio or audiovisual) and output mode (original speech or avatar with original speech) are independent. Audio input must not request a camera. Audio output must bypass avatar inference. Avatar listening motion uses silence as driving input only. Keep speech and interruption ahead of disposable idle animation. Understanding, original speech, subtitles, and avatar generation communicate through bounded asynchronous stages; compile and prewarm supported model paths. Keep browser control and video traffic on separate WebSockets. Show actual device assignments from health data. In the separate voice configuration, the input processing metric covers understanding and text decisions; it excludes asynchronous speech synthesis.

Display user ASR and capture timestamps, response generation/playback timestamps, and live microphone loudness plus server receipt feedback. Whisper subtitles run independently on the configured ASR GPU (default GPU 3); they do not replace MiniCPM audio input. Recover from temporary inference backlog with bounded capture buffers, conservative silence coalescing, and a visible pause/resume state. Preserve at least 480 ms of silence-driven avatar postroll, rounded up to complete A2V blocks, while keeping original speech samples intact.

For audiovisual input, use the official Streaming Omni Conversation prefix with explicit camera context and ground answers in the latest visible frame. Keep the audio-only prompt independent. Distinguish camera preview, server receipt and confirmed model vision processing. Confirm processing only from MiniCPM's actual vision slices/tokens for the entire input batch; show the frame count and camera capture time beside the preview. Runtime details include the exact most recently confirmed input image (the last frame of its batch), retaining at most 64 pending frame references and clearing them at session boundaries. Do not upload a paused, muted or ended camera's stale frame. Preview the complete camera view; runtime snapshots show the unmirrored image used by the model.

Capture camera input independently at 5 FPS (every 200 ms). Preserve the negotiated audio clock: normally 4–5 frames per 960 ms FlashHead block, or 5 per 1,000 ms voice-output block. Keep ordered individual capture timestamps and at most 6 queued snapshots to tolerate callback jitter. Skip missed slots after browser stalls; never fabricate history by duplicating the current image. Use MiniCPM's batched vision embedding feed and prewarm expected shapes. Bound audiovisual context with the model's native context-preserving window: 24 recent input units plus up to 500 tokens of older generated response text. Retain the system prompt. This limits raw audiovisual recall to the recent window; pure audio sessions restore their original window configuration.

Negotiate capture and MiniCPM decision duration from the active A2V worker's output frames and FPS. FlashHead Lite uses 24 frames at 25 FPS: 960 ms, 15,360 input samples and 23,040 driving samples. MiniCPM TTS retains its native output; a lossless sample buffer supplies exact A2V blocks. Audio output without an avatar retains a 1,000 ms input unit.

Use independent H.264 keyframe blocks when WebCodecs is supported, with JPEG fallback. Keep every frame and preserve the original frame indices for interruption anchors. Preserve original float32 speech separately from video encoding. Adapt upload credits to measured RTT (2–8 blocks); require browser media decode acknowledgements and bound unconfirmed delivery (2–6 blocks). Show queued input separately from ordinary unconfirmed transport and distinguish upload waiting from model processing. Stop must wait for an in-flight understanding step and submitted speech before clearing model state. Disable session start until the previous session has finished cleanup; show that cleanup is still in progress.

Interruption stops sound immediately. Retain up to four nearby frames for a visual deceleration of at most 250 ms, then continue silent driving from that precise frame's motion context. The server keeps a bounded cache of ten chunks so inference running ahead does not dictate the visible restart pose. Use a 120 ms blend into new frames; reduced motion omits the visual deceleration and uses an 80 ms blend. Keep interruption independent of the block clock.

Listening and interruption transitions use a monotonic visual clock independent of the playback AudioContext; speech video remains synchronized to its original audio. Prefetch one silent A2V block using browser playback feedback only after the response ends and the speech sample buffer is empty. Never advance A2V motion with disposable silence between speech blocks. Idle decoding must wait until preceding speech/postroll blocks have reserved their playback timeline. During actual delivery gaps with all video queues empty for at least 120 ms, reuse only already displayed silent frames from the current epoch, at most 24 frames and three seconds. Clear this cache on interruption/end, do not create audio from it, and disable this fallback for reduced motion. Smooth ordinary block boundaries over 80 ms and listening/speaking switches over 120 ms without changing original audio timing. Encode H.264 within the existing PyAV worker environment, retaining the FFmpeg subprocess compatibility path when PyAV is absent.

## Evidence on Hand

Existing model environments and deployment records in the parent workspace. Performance claims require new measured results.

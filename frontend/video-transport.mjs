export async function preferredVideoCodec() {
  try {
    if (typeof VideoDecoder === 'function' && (await VideoDecoder.isConfigSupported({
      codec: 'avc1.42C01E', codedWidth: 512, codedHeight: 512,
    })).supported) return 'h264';
  } catch {}
  return 'jpeg';
}

export async function decodeVideoBlock(message) {
  if (!message.video) {
    return Promise.all((message.frames || []).map(async data => {
      const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
      return createImageBitmap(new Blob([bytes], {type: 'image/jpeg'}));
    }));
  }
  const video = message.video, frames = [];
  let failure;
  const decoder = new VideoDecoder({output: frame => frames.push(frame), error: error => { failure = error; }});
  try {
    decoder.configure({codec: video.codec, codedWidth: message.width, codedHeight: message.height});
    for (let index = 0; index < video.chunks.length; index++) {
      decoder.decode(new EncodedVideoChunk({type: index === 0 ? 'key' : 'delta',
        timestamp: Math.round(index * 1e6 / message.fps),
        data: Uint8Array.from(atob(video.chunks[index]), c => c.charCodeAt(0))}));
    }
    await decoder.flush();
    if (failure) throw failure;
    if (frames.length !== video.frame_count) throw new Error('视频解码不完整，请重新开始会话。');
    return frames;
  } catch (error) {
    frames.forEach(frame => frame.close());
    throw error;
  } finally { if (decoder.state !== 'closed') decoder.close(); }
}

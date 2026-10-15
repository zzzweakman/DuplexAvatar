// A bounded visual deceleration; audio is stopped independently and immediately.
export function bridgePlan(frames, now, epoch, reducedMotion = false) {
  const chosen = reducedMotion ? [] : frames.filter(frame => frame.time >= now - .02 && frame.time <= now + .18).slice(0, 4);
  const last = chosen.at(-1);
  return {
    original: chosen,
    frames: chosen.map((frame, i) => ({...frame, epoch, bridge: true, time: now + .04 * (i + 1) + .01 * i * i})),
    anchor: last ? {sequence: last.sequence, frame_index: last.frame_index} : null,
  };
}

export function blendWeight(elapsed, duration) {
  const t = Math.max(0, Math.min(1, elapsed / duration));
  return 1 - Math.pow(1 - t, 3);
}

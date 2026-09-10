// Frame-by-frame video export: render one frame per output frame, stamp each
// with its own timestamp, encode with WebCodecs and mux to WebM. The output
// frame rate is free — it is NOT tied to the replay's 30 ticks/s, since the
// viewer interpolates between ticks, so 60 or 120 fps genuinely smooths the
// motion rather than duplicating frames.
// Nothing here is paced by wall clock, so a machine that cannot render 30 fps
// live produces the same file as one that can — it just takes longer.
//
// This exists because canvas.captureStream + MediaRecorder cannot be fixed by
// tuning. MediaRecorder timestamps frames by their ARRIVAL time: feed it
// frames slower than real time and it does not drop them, it stretches the
// video. And capturing a live audio graph on a machine that is stalling gives
// choppy audio because choppy audio is genuinely what was played. Both are
// structural. Here the video timeline is arithmetic (frame i is at i/fps s)
// and the audio is copied out of the decoded song by array slice, so neither
// can drift no matter how slow the render is.
import { Muxer, ArrayBufferTarget } from "webm-muxer";

export const CAN_EXPORT_EXACT =
  typeof VideoEncoder !== "undefined" && typeof AudioEncoder !== "undefined";

const RATE = 48000;  // Opus is natively 48 kHz
const ACHUNK = 960;  // 20 ms of audio per AudioData
const US = 1e6;

/** Decode a song blob to an AudioBuffer at RATE. decodeAudioData resamples
 * to the context's rate, so the slice below is always in 48 kHz frames. */
export async function decodeSong(blob) {
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!OAC) return null;
  const bytes = await blob.arrayBuffer();
  return await new OAC(2, RATE, RATE).decodeAudioData(bytes);
}

/** The song under [0, seconds) of the export as planar float channels.
 * `start` is the song time at frame 0; it can be negative (the recording
 * begins before the song does) or run past the end, and both are padded with
 * silence — so A/V alignment here is arithmetic, not luck. */
function audioSlice(buf, start, seconds) {
  const n = Math.ceil(seconds * RATE);
  const ch = Math.min(2, buf.numberOfChannels);
  const chans = [];
  const from0 = Math.round(start * RATE);
  for (let c = 0; c < ch; c++) {
    const dst = new Float32Array(n); // silence everywhere it is not covered
    const src = buf.getChannelData(c);
    const from = Math.max(0, from0), to = Math.min(src.length, from0 + n);
    if (to > from) dst.set(src.subarray(from, to), from - from0);
    chans.push(dst);
  }
  return { chans, n, ch };
}

/** Resolve once the encoder's queue has drained below `max`. */
function drain(enc, max) {
  if (enc.encodeQueueSize <= max) return null;
  return new Promise((res) => {
    const on = () => {
      if (enc.encodeQueueSize > max) return;
      enc.removeEventListener("dequeue", on);
      res();
    };
    enc.addEventListener("dequeue", on);
  });
}

/** Render `frames` frames at `fps` through `renderFrame(i)` and return a WebM
 * Blob. `renderFrame` must leave frame i in `canvas` synchronously — mapping
 * a frame index onto a replay position is the caller's business, so nothing
 * here needs to know the tick rate. `song` is a decoded AudioBuffer (or null)
 * and `songStart` the song time at frame 0. */
export async function exportExact({
  canvas, frames, fps = 30, renderFrame, song = null, songStart = 0,
  onProgress = () => {},
}) {
  // Even dimensions: VP9 4:2:0 needs them, and the live canvas is whatever
  // size the window happens to be. Copying through a 2D canvas also pins the
  // output size for the whole export, so a resize mid-export cannot corrupt
  // the stream.
  const W = Math.max(2, canvas.width - (canvas.width % 2));
  const H = Math.max(2, canvas.height - (canvas.height % 2));
  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const g2d = off.getContext("2d", { alpha: false });

  const audio = song ? audioSlice(song, songStart, frames / fps) : null;

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: "V_VP9", width: W, height: H, frameRate: fps },
    ...(audio
      ? { audio: { codec: "A_OPUS", numberOfChannels: audio.ch,
                   sampleRate: RATE } }
      : {}),
  });

  let failed = null;
  const vEnc = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => { failed ??= e; },
  });
  vEnc.configure({
    codec: "vp09.00.10.08", // profile 0, level 1.0, 8-bit
    width: W,
    height: H,
    // more frames want more bits, but not proportionally — consecutive
    // frames are highly correlated, so the extra ones code cheaply
    bitrate: Math.round(8_000_000 * Math.sqrt(fps / 30)),
    framerate: fps,
    latencyMode: "quality",
  });

  for (let i = 0; i < frames; i++) {
    if (failed) break;
    renderFrame(i);
    // read the WebGL canvas in the SAME task as the render: the drawing
    // buffer is not preserved across a compositing step
    g2d.drawImage(canvas, 0, 0, W, H);
    const vf = new VideoFrame(off, {
      timestamp: Math.round((i * US) / fps),
      duration: Math.round(US / fps),
    });
    vEnc.encode(vf, { keyFrame: i % Math.round(fps * 2) === 0 });
    vf.close();
    const wait = drain(vEnc, 8);
    if (wait) await wait;          // also yields, so the UI stays alive
    if (i % 10 === 0) onProgress(i / frames);
  }

  if (audio && !failed) {
    const aEnc = new AudioEncoder({
      output: (chunk, meta) => muxer.addAudioChunk(chunk, meta),
      error: (e) => { failed ??= e; },
    });
    aEnc.configure({
      codec: "opus", sampleRate: RATE,
      numberOfChannels: audio.ch, bitrate: 128_000,
    });
    for (let o = 0; o < audio.n && !failed; o += ACHUNK) {
      const len = Math.min(ACHUNK, audio.n - o);
      const planar = new Float32Array(len * audio.ch);
      for (let c = 0; c < audio.ch; c++)
        planar.set(audio.chans[c].subarray(o, o + len), c * len);
      const ad = new AudioData({
        format: "f32-planar",
        sampleRate: RATE,
        numberOfFrames: len,
        numberOfChannels: audio.ch,
        timestamp: Math.round((o * US) / RATE),
        data: planar,
      });
      aEnc.encode(ad);
      ad.close();
      const wait = drain(aEnc, 16);
      if (wait) await wait;
    }
    await aEnc.flush();
    aEnc.close();
  }

  await vEnc.flush();
  vEnc.close();
  if (failed) throw failed;
  muxer.finalize();
  onProgress(1);
  return new Blob([target.buffer], { type: "video/webm" });
}

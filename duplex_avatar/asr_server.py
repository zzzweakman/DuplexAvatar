"""Independent subtitle worker, placed by the manager on the configured ASR GPU."""
import sys

from duplex_avatar.settings import CONFIG, ROOT, WORKSPACE

sys.path.insert(0, str(WORKSPACE / "Omni-Interaction-Agent/minicpm_ft"))
from mcpmft.infer.asr import AsrSettings, FasterWhisperTranscriber, create_app


if __name__ == "__main__":
    import soundfile as sf
    import numpy as np
    import uvicorn

    settings = AsrSettings(model_path=str(WORKSPACE / CONFIG["asr_model"]), beam_size=1)
    transcriber = FasterWhisperTranscriber(settings)
    # Complete CUDA/VAD cold initialization before reporting ready.
    fixture = ROOT / "runtime/fixtures/question.wav"
    if fixture.exists():
        audio, rate = sf.read(fixture, dtype="float32")
        if rate != 16000:
            from scipy.signal import resample_poly
            import math
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        print("ASR warmup:", transcriber.transcribe(pcm, start_ms=0, language="zh")["text"], flush=True)
    app = create_app(settings, transcriber)
    uvicorn.run(app, host="127.0.0.1", port=CONFIG["asr_port"])

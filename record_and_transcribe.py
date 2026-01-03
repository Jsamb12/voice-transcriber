import sounddevice as sd
from scipy.io.wavfile import write
import whisper

SAMPLE_RATE = 16_000
SECONDS = 5
AUDIO_FILE = "input.wav"

print("Recording... Speak now.")
audio = sd.rec(
    int(SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)
sd.wait()
write(AUDIO_FILE, SAMPLE_RATE, audio)
print("Recording saved.")

print("Loading whisper model.")
model = whisper.load_model("base", device="cpu")

print("Transcribing")
result = model.transcribe(AUDIO_FILE, fp16=False)

print("\nTranscription:")
print(result["text"].strip())
